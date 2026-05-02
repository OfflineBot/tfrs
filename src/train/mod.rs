pub mod data;
pub mod mail;
pub mod train;
pub mod train_mail;
pub use train::{TrainConfig, overfit_one_batch, train_copy_task};
pub use train_mail::{MailMode, MailTrainConfig, TrainBudget, train_and_eval_mail};
