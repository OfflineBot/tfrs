mod model;
mod train;
mod utils;

use std::path::PathBuf;

use crate::train::{
    MailMode, MailTrainConfig, TrainBudget, train_and_eval_mail,
};

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "mail".into());
    match mode.as_str() {
        // ---------- email classification ----------
        // Hier wird ALLES konfiguriert. Anpassen und neu kompilieren.
        "mail" => {
            let cfg = MailTrainConfig {
                // ===== was soll passieren? =====
                //   MailMode::Fresh     -> neu initialisieren, trainieren, speichern
                //   MailMode::Continue  -> aus save_path laden, weitertrainieren, speichern
                //   MailMode::EvalOnly  -> aus save_path laden, nur testen (kein training)
                mode: MailMode::Fresh,

                // ===== wie viel trainieren? =====
                //   TrainBudget::Steps(N)  -> genau N Gradient-Updates
                //   TrainBudget::Epochs(N) -> N volle Durchläufe übers Train-Set
                budget: TrainBudget::Steps(20_000),

                // ===== Hyperparameter =====
                d_model:      64,
                n_heads:      4,
                d_ff:         128,
                n_enc_layers: 2,
                seq_len:      512,
                lr:           1e-3,
                seed:         0xC0FFEE,
                log_every:    50,

                // ===== Checkpoint =====
                save_path: PathBuf::from("mail_model.bin"),

                // ===== Datensätze =====
                // leer = automatisch ~/Downloads/dataset*.ron einlesen
                files: Vec::new(),
            };
            train_and_eval_mail(cfg);
        }

        other => {
            eprintln!("usage: transformer [copy|mail]   (got {other:?})");
            std::process::exit(2);
        }
    }
}
