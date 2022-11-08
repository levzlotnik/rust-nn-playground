pub mod tensor_accessor;
pub mod tensor_trait;

use num_traits::Num;

/// A trait for any eligible data type for tensors.
pub trait Dtype: Num {
    fn cast<To: Num + From<Self>>(self) -> To {
        To::from(self)
    }
}