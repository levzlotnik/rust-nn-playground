use std::ops::Index;

use super::{Dtype, tensor_accessor::TensorAccessor};

pub trait TensorTrait<T: Dtype> {
    type Accessor: TensorAccessor<T>;

    /// Gets the immutable accessor to the current tensor
    fn get_accessor<'a>(&'a self) -> &'a Self::Accessor;

    fn get_mut_accessor<'a>(&'a mut self) -> &'a mut Self::Accessor;
}

impl<T: Dtype, TensorImpl> Index<i64> for TensorImpl 
    where TensorImpl: TensorTrait<T>
{

    fn index(&self, index: i64) -> &Self::Output {
        
    }

}