use std::ops::Range;

pub type SizeHint = (usize, Option<usize>);

#[inline]
pub fn size_hint_add(a: SizeHint, b: SizeHint) -> SizeHint {
    let min = a.0.saturating_add(b.0);
    let max = match (a.1, b.1) {
        (Some(x), Some(y)) => x.checked_add(y),
        _ => None,
    };

    (min, max)
}

#[inline]
pub fn size_hint_mul(a: SizeHint, b: SizeHint) -> SizeHint {
    let low = a.0.saturating_mul(b.0);
    let hi = match (a.1, b.1) {
        (Some(x), Some(y)) => x.checked_mul(y),
        (Some(0), None) | (None, Some(0)) => Some(0),
        _ => None,
    };

    (low, hi)
}

#[derive(Clone)]
struct MultiProductIter {
    cur: Option<usize>,
    iter: Range<usize>,
    iter_orig: Range<usize>,
}

impl MultiProductIter {
    fn iterate(&mut self) { self.cur = self.iter.next(); }

    fn reset(&mut self) { self.iter = self.iter_orig.clone(); }

    fn in_progress(&self) -> bool { self.cur.is_some() }
}

enum MultiProductIterState {
    StartOfIter,
    MidIter { on_first_iter: bool },
}

#[derive(Clone)]
pub struct MultiProduct<const DIM: usize>([MultiProductIter; DIM]);

impl<const DIM: usize> MultiProduct<DIM> {
    pub fn new(sizes: [usize; DIM]) -> Self {
        MultiProduct(sizes.map(|s| MultiProductIter {
            cur: None,
            iter: 0..s,
            iter_orig: 0..s,
        }))
    }

    fn iterate_last(
        multi_iters: &mut [MultiProductIter],
        mut state: MultiProductIterState,
    ) -> bool {
        use MultiProductIterState::*;

        if let Some((last, rest)) = multi_iters.split_last_mut() {
            let on_first_iter = match state {
                StartOfIter => {
                    let on_first_iter = !last.in_progress();
                    state = MidIter { on_first_iter };
                    on_first_iter
                },
                MidIter { on_first_iter } => on_first_iter,
            };

            if !on_first_iter {
                last.iterate();
            }

            if last.in_progress() {
                true
            } else if MultiProduct::<DIM>::iterate_last(rest, state) {
                last.reset();
                last.iterate();
                // If iterator is None twice consecutively, then iterator is
                // empty; whole product is empty.
                last.in_progress()
            } else {
                false
            }
        } else {
            // Reached end of iterator list. On initialisation, return true.
            // At end of iteration (final iterator finishes), finish.
            match state {
                StartOfIter => false,
                MidIter { on_first_iter } => on_first_iter,
            }
        }
    }

    /// Returns the unwrapped value of the next iteration.
    fn curr_iterator(&self) -> [usize; DIM] {
        array_init::array_init(|i| self.0[i].cur.clone().unwrap())
    }

    /// Returns true if iteration has started and has not yet finished; false
    /// otherwise.
    fn in_progress(&self) -> bool {
        if let Some(last) = self.0.last() {
            last.in_progress()
        } else {
            false
        }
    }
}

impl<const DIM: usize> Iterator for MultiProduct<DIM> {
    type Item = [usize; DIM];

    fn next(&mut self) -> Option<Self::Item> {
        if MultiProduct::<DIM>::iterate_last(&mut self.0, MultiProductIterState::StartOfIter) {
            Some(self.curr_iterator())
        } else {
            None
        }
    }

    fn count(self) -> usize {
        if self.0.is_empty() {
            return 0;
        }

        if !self.in_progress() {
            return IntoIterator::into_iter(self.0)
                .fold(1, |acc, multi_iter| acc * multi_iter.iter.clone().count());
        }

        IntoIterator::into_iter(self.0).fold(
            0,
            |acc,
             MultiProductIter {
                 iter,
                 iter_orig,
                 cur: _,
             }| {
                let total_count = iter_orig.clone().count();
                let cur_count = iter.clone().count();
                acc * total_count + cur_count
            },
        )
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Not ExactSizeIterator because size may be larger than usize
        if self.0.is_empty() {
            return (0, Some(0));
        }

        if !self.in_progress() {
            return self.0.iter().fold((1, Some(1)), |acc, multi_iter| {
                size_hint_mul(acc, multi_iter.iter.size_hint())
            });
        }

        self.0.iter().fold(
            (0, Some(0)),
            |acc,
             &MultiProductIter {
                 ref iter,
                 ref iter_orig,
                 cur: _,
             }| {
                let cur_size = iter.size_hint();
                let total_size = iter_orig.size_hint();

                size_hint_add(size_hint_mul(acc, total_size), cur_size)
            },
        )
    }
}
