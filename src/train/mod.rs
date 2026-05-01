pub mod data;
pub mod train;
pub use train::{TrainConfig, overfit_one_batch, train_copy_task};
