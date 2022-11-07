use std::ops::{Range, RangeFrom, RangeTo, RangeFull, Index, IndexMut};

use num_traits::Num;

/// A trait for any eligible data type for tensors.
pub trait Dtype: Num {
    fn cast<To: Num + From<Self>>(self) -> To {
        To::from(self)
    }
}

trait MultiIndexBundleBase<'a> {
    type ScalarSlice;
    type RangeSlice;
}


trait TensorAccessor<'a, T: Dtype, MultiIndexBundle: MultiIndexBundleBase<'a>>
{
    /// Enables accessing a slice of the tensor with a scalar
    fn index_scalar(&self, pos: i64) -> &'a MultiIndexBundle::ScalarSlice;

    /// Enables accessing a mutable slice of the tensor with a scalar
    fn index_scalar_mut(&mut self, pos: i64) -> &'a mut MultiIndexBundle::ScalarSlice;

    /// Enables accessing a slice of the tensor with a range
    fn index_range(&self, range: Range<i64>) -> &'a MultiIndexBundle::RangeSlice;

    /// Enables accessing a mutable slice of the tensor with a range
    fn index_range_mut(&mut self, range: Range<i64>) -> &'a mut MultiIndexBundle::RangeSlice;

    fn get_dim_size(&self) -> i64;
} 

impl<'a, T: Dtype, MultiIndexBundle> Index<i64> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    type Output = MultiIndexBundle::ScalarSlice;

    fn index(&self, index: i64) -> &'a Self::Output {
        &*self.index_scalar(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> IndexMut<i64> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    fn index_mut(&mut self, index: i64) -> &'a mut Self::Output {
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

impl<'a, T: Dtype, MultiIndexBundle> Index<Range<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    type Output = MultiIndexBundle::RangeSlice;

    fn index(&self, index: Range<i64>) -> &'a Self::Output {
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> IndexMut<Range<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    fn index_mut(&mut self, index: Range<i64>) -> &'a mut Self::Output {
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> Index<RangeFrom<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    type Output = MultiIndexBundle::RangeSlice;

    fn index(&self, index: RangeFrom<i64>) -> &'a Self::Output {
        let index = Range {
            start: index.start,
            end: self.get_dim_size()
        };
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> IndexMut<RangeFrom<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    fn index_mut(&mut self, index: RangeFrom<i64>) -> &'a mut Self::Output {
        let index = Range {
            start: index.start,
            end: self.get_dim_size()
        };
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}


impl<'a, T: Dtype, MultiIndexBundle> Index<RangeTo<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    type Output = MultiIndexBundle::RangeSlice;

    fn index(&self, index: RangeTo<i64>) -> &'a Self::Output {
        let index = Range {
            start: 0,
            end: index.end
        };
        let index = normalize_range(index, self.get_dim_size());
        &*self.index_range(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> IndexMut<RangeTo<i64>> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    fn index_mut(&mut self, index: RangeTo<i64>) -> &'a mut Self::Output {
        let index = Range {
            start: 0,
            end: index.end
        };
        let index = normalize_range(index, self.get_dim_size());
        &mut *self.index_range_mut(index)
    }
}


impl<'a, T: Dtype, MultiIndexBundle> Index<RangeFull> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    type Output = MultiIndexBundle::RangeSlice;

    fn index(&self, _: RangeFull) -> &'a Self::Output {
        let index = Range {
            start: 0,
            end: self.get_dim_size()
        };
        &*self.index_range(index)
    }
}

impl<'a, T: Dtype, MultiIndexBundle> IndexMut<RangeFull> for dyn TensorAccessor<'a, T, MultiIndexBundle> 
    where MultiIndexBundle: MultiIndexBundleBase<'a>
{
    fn index_mut(&mut self, _: RangeFull) -> &'a mut Self::Output {
        let index = Range {
            start: 0,
            end: self.get_dim_size()
        };
        &mut *self.index_range_mut(index)
    }
}
