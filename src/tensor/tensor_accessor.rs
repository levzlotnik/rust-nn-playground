use std::ops::{Range, RangeFrom, RangeTo, RangeFull, Index, IndexMut};

use super::Dtype;
use super::tensor_trait::TensorTrait;


pub trait MultiIndexBundleBase<T: Dtype> {
    type ScalarSlice: TensorTrait<T>;
    type RangeSlice: TensorTrait<T>;
}


pub trait TensorAccessor<T: Dtype>
{
    type MultiIndexBundle: MultiIndexBundleBase<T>;

    /// Enables accessing a slice of the tensor with a scalar
    fn index_scalar<'a>(&'a self, pos: i64) -> &'a <Self::MultiIndexBundle as MultiIndexBundleBase<T>>::ScalarSlice;

    /// Enables accessing a mutable slice of the tensor with a scalar
    fn index_scalar_mut<'a>(&'a mut self, pos: i64) -> &'a mut <Self::MultiIndexBundle as MultiIndexBundleBase<T>>::ScalarSlice;

    /// Enables accessing a slice of the tensor with a range
    fn index_range<'a>(&'a self, range: Range<i64>) -> &'a <Self::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    /// Enables accessing a mutable slice of the tensor with a range
    fn index_range_mut<'a>(&'a mut self, range: Range<i64>) -> &'a mut <Self::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    fn get_dim_size(&self) -> i64;
} 

impl<T: Dtype, TA: TensorAccessor<T>> Index<i64> for TA 
{
    type Output = <TA::MultiIndexBundle as MultiIndexBundleBase<T>>::ScalarSlice;

    fn index<'a>(&'a self, index: i64) -> &'a Self::Output {
        &*self.index_scalar(index)
    }
}

impl<T: Dtype, TA:TensorAccessor<T>> IndexMut<i64> for TA
{
    fn index_mut<'a>(&'a mut self, index: i64) -> &'a mut Self::Output {
        &mut *self.index_scalar_mut(index)
    }
}

fn normalize_range(range: Range<i64>, size: i64) -> Range<i64> {
    let Range {start, end} = range;
    if start < -size || start >= size {
        panic!("index out of bounds: the size {} allows for indices between [{}, {}]", size, -size, size-1);
    }
    if end < -size || end > size {
        panic!("index out of bounds: the size {} allows for indices between [{}, {}]", size, -size, size);
    }

    let start = (start + size) % size;
    let end = (end + size) % (size + 1);
    Range {start, end}
}

impl<T: Dtype, TA:TensorAccessor<T>> Index<Range<i64>> for TA
{
    type Output = <TA::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    fn index<'a>(&'a self, index: Range<i64>) -> &'a Self::Output {
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<T: Dtype, TA> IndexMut<Range<i64>> for TA
    where TA: TensorAccessor<T>,
{
    fn index_mut<'a>(&'a mut self, index: Range<i64>) -> &'a mut Self::Output {
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}

impl<T: Dtype, TA: TensorAccessor<T>> Index<RangeFrom<i64>> for TA
{
    type Output = <TA::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    fn index<'a>(&'a self, index: RangeFrom<i64>) -> &'a Self::Output {
        let index = Range {
            start: index.start,
            end: self.get_dim_size()
        };
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<T: Dtype, TA: TensorAccessor<T>> IndexMut<RangeFrom<i64>> for TA
{
    fn index_mut<'a>(&'a mut self, index: RangeFrom<i64>) -> &'a mut Self::Output {
        let index = Range {
            start: index.start,
            end: self.get_dim_size()
        };
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}


impl<T: Dtype, TA: TensorAccessor<T>> Index<RangeTo<i64>> for TA
{
    type Output = <TA::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    fn index<'a>(&'a self, index: RangeTo<i64>) -> &'a Self::Output {
        let index = Range {
            start: 0,
            end: index.end
        };
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<T: Dtype, TA: TensorAccessor<T>> IndexMut<RangeTo<i64>> for TA
{
    fn index_mut<'a>(&'a mut self, index: RangeTo<i64>) -> &'a mut Self::Output {
        let index = Range {
            start: 0,
            end: index.end
        };
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}


impl<T: Dtype, TA: TensorAccessor<T>> Index<RangeFull> for TA
{
    type Output = <TA::MultiIndexBundle as MultiIndexBundleBase<T>>::RangeSlice;

    fn index<'a>(&'a self, _: RangeFull) -> &'a Self::Output {
        let index = Range {
            start: 0,
            end: self.get_dim_size()
        };
        &*self.index_range(index)
    }
}

impl<T: Dtype, TA: TensorAccessor<T>> IndexMut<RangeFull> for TA
{
    fn index_mut<'a>(&'a mut self, _: RangeFull) -> &'a mut Self::Output {
        let index = Range {
            start: 0,
            end: self.get_dim_size()
        };
        &mut *self.index_range_mut(index)
    }
}
