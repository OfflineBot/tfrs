mod model;
mod train;
mod utils;

use std::path::PathBuf;
use std::process::ExitCode;

use crate::train::{
    MailInput, MailMode, MailTrainConfig, PredictConfig, StopCriteria, train_and_eval_mail,
    predict_one,
};

const DEFAULT_MODEL: &str = "bin/mail_model.bin";

fn print_usage() {
    eprintln!(
        r#"tfrs — encoder-only transformer for multi-label mail classification

USAGE:
    tfrs train     [OPTIONS]                     train from scratch
    tfrs continue  [OPTIONS]                     continue training a checkpoint
    tfrs resume    [OPTIONS]                     alias for `continue`
    tfrs eval      [OPTIONS]                     evaluate a checkpoint on the test split
    tfrs predict   [PREDICT OPTIONS]             predict labels for a single email
    tfrs --predict "<text>"                      shortcut for `predict --text "<text>"`

COMMON OPTIONS:
    --model PATH                checkpoint file (default: {DEFAULT_MODEL})
    --seq-len N                 token window (default: 512)
    --d-model N                 model width  (default: 64)
    --n-heads N                 attention heads (default: 4)
    --d-ff N                    FFN width  (default: 128)
    --n-enc-layers N            encoder layers (default: 2)

TRAIN / CONTINUE / EVAL OPTIONS:
    --files PATH [PATH ...]     dataset file(s); default: scan ~/Downloads/dataset*.ron
    --lr F                      learning rate (default: 1e-3)
    --seed N                    PRNG seed
    --log-every N               training log cadence (default: 50)

STOPPING CRITERIA (combine freely; first to fire stops training):
    --max-steps N               cap on gradient updates
    --max-epochs N              cap on full passes over the train set
    --target-loss F             stop when test loss <= F
    --target-label-acc F        stop when label accuracy >= F (e.g. 0.95)
    --target-exact-acc F        stop when exact-match accuracy >= F
    --eval-every N              also evaluate criteria every N steps mid-epoch

PREDICT OPTIONS:
    --text "..."                full email text. Best results with format:
                                "<subject> | <sender_name> <<sender_email>> | <body>"
    --subject "..."             subject line     (optional)
    --from "..."                sender name      (optional)
    --sender-email "..."        sender address   (optional)
    --body "..."                body text        (required if --text not given)
    --threshold F               positive-label cutoff (default: 0.5)

EXAMPLES:
    # train until 95% label accuracy OR 20k steps, whichever first
    tfrs train --max-steps 20000 --target-label-acc 0.95 --eval-every 500

    # quick predict on raw text
    tfrs --predict "Sehr geehrte Damen und Herren, anbei die Rechnung."

    # structured predict mirroring the training format
    tfrs predict --subject "Rechnung Nr. 42" \
                 --from "Acme GmbH" --sender-email "billing@acme.de" \
                 --body "Anbei die Rechnung als PDF."
"#,
        DEFAULT_MODEL = DEFAULT_MODEL
    );
}

struct Args {
    positional: Vec<String>,
    flags: std::collections::BTreeMap<String, Vec<String>>,
}

impl Args {
    fn parse(argv: &[String]) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut flags: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            if let Some(name) = a.strip_prefix("--") {
                // collect values until next --flag or end
                let key = name.to_string();
                let mut values = Vec::new();
                let mut j = i + 1;
                while j < argv.len() && !argv[j].starts_with("--") {
                    values.push(argv[j].clone());
                    j += 1;
                }
                flags.entry(key).or_default().extend(values);
                i = j;
            } else {
                positional.push(a.clone());
                i += 1;
            }
        }
        Ok(Self { positional, flags })
    }

    fn first_value(&self, key: &str) -> Option<&str> {
        self.flags.get(key).and_then(|v| v.first()).map(|s| s.as_str())
    }
    fn values(&self, key: &str) -> Option<&[String]> {
        self.flags.get(key).map(|v| v.as_slice())
    }
    fn parse_opt<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.first_value(key) {
            None => Ok(None),
            Some(s) => s.parse::<T>().map(Some).map_err(|_| format!("--{key}: cannot parse {s:?}")),
        }
    }
    fn has(&self, key: &str) -> bool { self.flags.contains_key(key) }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    let args = Args::parse(&argv)?;

    // shortcut: `tfrs --predict "text"` → predict subcommand with --text
    if args.positional.is_empty() && args.has("predict") {
        let text = args.first_value("predict").unwrap_or("").to_string();
        let cfg = build_predict_cfg(&args)?;
        return run_predict(cfg, MailInput::from_body(text), &args);
    }

    let sub = args.positional.first().map(|s| s.as_str()).unwrap_or("");
    match sub {
        "train"              => run_train(args, MailMode::Fresh),
        "continue" | "resume" => run_train(args, MailMode::Continue),
        "eval"               => run_train(args, MailMode::EvalOnly),
        "predict"  => {
            let cfg = build_predict_cfg(&args)?;
            let input = build_predict_input(&args)?;
            run_predict(cfg, input, &args)
        }
        "" => { print_usage(); Err("missing subcommand".into()) }
        other => { print_usage(); Err(format!("unknown subcommand: {other:?}")) }
    }
}

fn arch_defaults(args: &Args) -> Result<(usize, usize, usize, usize, usize), String> {
    Ok((
        args.parse_opt::<usize>("d-model")?.unwrap_or(64),
        args.parse_opt::<usize>("n-heads")?.unwrap_or(4),
        args.parse_opt::<usize>("d-ff")?.unwrap_or(128),
        args.parse_opt::<usize>("n-enc-layers")?.unwrap_or(2),
        args.parse_opt::<usize>("seq-len")?.unwrap_or(512),
    ))
}

fn run_train(args: Args, mode: MailMode) -> Result<(), String> {
    let (d_model, n_heads, d_ff, n_enc_layers, seq_len) = arch_defaults(&args)?;
    let save_path = args.first_value("model").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let files: Vec<PathBuf> = args.values("files")
        .map(|v| v.iter().map(PathBuf::from).collect())
        .unwrap_or_default();

    let stop = StopCriteria {
        max_steps:        args.parse_opt::<usize>("max-steps")?,
        max_epochs:       args.parse_opt::<usize>("max-epochs")?,
        target_loss:      args.parse_opt::<f32>("target-loss")?,
        target_label_acc: args.parse_opt::<f32>("target-label-acc")?,
        target_exact_acc: args.parse_opt::<f32>("target-exact-acc")?,
        eval_every_steps: args.parse_opt::<usize>("eval-every")?,
    };

    let cfg = MailTrainConfig {
        mode,
        stop,
        d_model, n_heads, d_ff, n_enc_layers, seq_len,
        lr:        args.parse_opt::<f32>("lr")?.unwrap_or(1e-3),
        seed:      args.parse_opt::<u64>("seed")?.unwrap_or(0xC0FFEE),
        log_every: args.parse_opt::<usize>("log-every")?.unwrap_or(50),
        save_path,
        files,
    };
    train_and_eval_mail(cfg);
    Ok(())
}

fn build_predict_cfg(args: &Args) -> Result<PredictConfig, String> {
    let (d_model, n_heads, d_ff, n_enc_layers, seq_len) = arch_defaults(args)?;
    let model_path = args.first_value("model").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    Ok(PredictConfig {
        model_path,
        seq_len, d_model, n_heads, d_ff, n_enc_layers,
        threshold: args.parse_opt::<f32>("threshold")?.unwrap_or(0.5),
        lr:        args.parse_opt::<f32>("lr")?.unwrap_or(1e-3),
    })
}

fn build_predict_input(args: &Args) -> Result<MailInput, String> {
    if let Some(text) = args.first_value("text") {
        return Ok(MailInput::from_body(text));
    }
    let subject      = args.first_value("subject").unwrap_or("").to_string();
    let sender_name  = args.first_value("from").unwrap_or("").to_string();
    let sender_email = args.first_value("sender-email").unwrap_or("").to_string();
    let body         = args.first_value("body").unwrap_or("").to_string();
    if subject.is_empty() && sender_name.is_empty() && sender_email.is_empty() && body.is_empty() {
        return Err("predict: provide --text, --body, or any of --subject/--from/--sender-email".into());
    }
    Ok(MailInput { subject, sender_name, sender_email, body })
}

fn run_predict(cfg: PredictConfig, input: MailInput, _args: &Args) -> Result<(), String> {
    if !cfg.model_path.exists() {
        return Err(format!(
            "model file not found: {} — train one first (e.g. `tfrs train --max-epochs 3`)",
            cfg.model_path.display()
        ));
    }
    println!("model: {}", cfg.model_path.display());
    let preds = predict_one(&cfg, &input);
    println!("--- prediction ---");
    let max_name = preds.iter().map(|p| p.name.len()).max().unwrap_or(0);
    for p in &preds {
        let mark = if p.predicted { "*" } else { " " };
        println!("  {mark} {:<width$}  {:>6.2}%  {}", p.name, p.probability * 100.0,
                 if p.predicted { "yes" } else { "no" },
                 width = max_name);
    }
    let positives: Vec<&str> = preds.iter().filter(|p| p.predicted).map(|p| p.name.as_str()).collect();
    println!("labels: {:?}", positives);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::from(2) }
    }
}
