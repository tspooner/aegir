//! Module for helper error types.

/// Placeholder for infallible operations.
///
/// This type is mostly used in conjunction with [BinaryError] for operators
/// that guarantee no errors within their own transformation code. This is
/// better than relying on `either` as the semantic meaning of the three error
/// cases is preserved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoError {}

impl std::fmt::Display for NoError {
    fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { match *self {} }
}

impl std::error::Error for NoError {
    fn description(&self) -> &str { match *self {} }
}

/// Error type for unary operations.
///
/// This type is typically used when an operator is defined over a single
/// sub-operator; e.g. negation or trigonometric functions. The
/// [Inner](UnaryError::Inner) case allows for error propagation from the
/// underlying operator, and unique errors arising within the given
/// operator itself via [Output](UnaryError::Output).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryError<I, O> {
    /// Inner operator failed.
    Inner(I),

    /// Error occurred post-hoc.
    Output(O),
}

impl<I, O> std::fmt::Display for UnaryError<I, O>
where
    I: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryError::Inner(e) => e.fmt(f),
            UnaryError::Output(e) => e.fmt(f),
        }
    }
}

impl<I, O> std::error::Error for UnaryError<I, O>
where
    I: std::fmt::Debug + std::fmt::Display,
    O: std::fmt::Debug + std::fmt::Display,
{
}

/// Error type for binary operations.
///
/// This type is typically used when an operator is defined over a pair of
/// sub-operators; e.g. addition, multiplication, subtraction. The
/// [Left](BinaryError::Left) and [Right](BinaryError::Right) cases then allow
/// for error propagation from said pair, as well as unique errors arising
/// within the given operator itself via [Output](BinaryError::Output).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryError<L, R, O> {
    /// Left operator failed.
    Left(L),

    /// Right operator failed.
    Right(R),

    /// Error occurred post-hoc.
    Output(O),
}

impl<L, R, O> std::fmt::Display for BinaryError<L, R, O>
where
    L: std::fmt::Display,
    R: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::Left(e) => e.fmt(f),
            BinaryError::Right(e) => e.fmt(f),
            BinaryError::Output(e) => e.fmt(f),
        }
    }
}

impl<L, R, O> std::error::Error for BinaryError<L, R, O>
where
    L: std::fmt::Debug + std::fmt::Display,
    R: std::fmt::Debug + std::fmt::Display,
    O: std::fmt::Debug + std::fmt::Display,
{
}
