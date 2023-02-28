use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Patient {
    pub demand: f32,
    pub start_time: f32,
    pub end_time: f32,
    pub care_time: f32,
    pub x_coord: f32,
    pub y_coord: f32,
}
