pub type TF = bool;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TernaryTruthValue {
    True,
    False,
    Unknown
}

pub type TFU = TernaryTruthValue;

impl PartialEq<bool> for TFU {
    fn eq(&self, rhs: &bool) -> bool {
        match self {
            // Yes, I know I could just return rhs or !rhs... be calm, this is more readable.
            TFU::True => *rhs == true,
            TFU::False => *rhs == false,
            _ => false,
        }
    }
}

impl std::ops::BitAnd for TFU {
    type Output = TFU;

    fn bitand(self, rhs: TFU) -> TFU {
        match (self, rhs) {
            // Boolean logic:
            (TFU::True, TFU::True) => TFU::True,
            (TFU::True, TFU::False) => TFU::False,
            (TFU::False, TFU::True) => TFU::False,
            (TFU::False, TFU::False) => TFU::False,

            // Ternary extension:
            (TFU::Unknown, _) => TFU::Unknown,
            (_, TFU::Unknown) => TFU::Unknown,
        }
    }
}

impl std::ops::BitOr for TFU {
    type Output = TFU;

    fn bitor(self, rhs: TFU) -> TFU {
        match (self, rhs) {
            // Boolean logic:
            (TFU::True, TFU::True) => TFU::True,
            (TFU::True, TFU::False) => TFU::True,
            (TFU::False, TFU::True) => TFU::True,
            (TFU::False, TFU::False) => TFU::False,

            // Ternary extension:
            (TFU::Unknown, _) => TFU::Unknown,
            (_, TFU::Unknown) => TFU::Unknown,
        }
    }
}

impl From<bool> for TFU {
    fn from(val: bool) -> TFU {
        match val {
            true => TFU::True,
            false => TFU::False,
        }
    }
}

impl std::fmt::Display for TFU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TFU::True => write!(f, "true"),
            TFU::False => write!(f, "false"),
            TFU::Unknown => write!(f, "?"),
        }
    }
}
