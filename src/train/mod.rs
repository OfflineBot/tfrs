pub mod data;
pub mod mail;
pub mod train;
pub mod train_mail;
pub use train::{TrainConfig, overfit_one_batch, train_copy_task};
pub use train_mail::{
    MailInput, MailMode, MailTrainConfig, PredictConfig, PredictedLabel, StopCriteria,
    TrainBudget, predict_one, train_and_eval_mail,
};
