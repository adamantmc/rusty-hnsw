use std::cmp::Ordering;
use std::fmt::{Display, Formatter};

#[derive(Copy, Clone, PartialEq)]
pub struct Dist(pub f32);

impl Display for Dist {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Eq for Dist {}

impl Ord for Dist {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd<Self> for Dist {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}