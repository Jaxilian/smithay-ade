// The caching logic is used to process surface synchronization. It creates
// an effective decoupling between the moment the client sends wl_surface.commit
// and the moment where the state that was committed is actually applied by the
// compositor.
//
// The way this is modelled in Smithay is through the `Cache` type, which is a container
// representing a cached state for a particular type. The full cached state of a surface
// is thus composed of a a set of `Cache<T>` for all relevant `T`, as modelled by the
// `MultiCache`.
//
// The logic of the `Cache` is as follows:
//
// - The protocol handlers mutably access the `pending` state to modify it accord to
//   the client requests
// - On commit, a snapshot of this pending state is created by invoking `Cacheable::commit`
//   and stored in the cache alongside an externally provided id
// - When the compositor decides that a given state (represented by its commit id) should
//   become active, `Cache::apply_state` is invoked with that commit id. The associated state
//   is then applied to the `current` state, that the compositor can then use as a reference
//   for the current window state. Note that, to preserve the commit ordering, all states
//   with a commit id older than the one requested are applied as well, in order.
//
// The logic for generating these commit ids and deciding when to apply them is implemented
// and described in `transaction.rs`.

use std::{
    collections::VecDeque,
    sync::{Mutex, MutexGuard},
};

use downcast_rs::{Downcast, impl_downcast};
use wayland_server::DisplayHandle;

use crate::utils::Serial;

/// Trait representing a value that can be used in double-buffered storage
///
/// The type needs to implement the [`Default`] trait, which will be used
/// to initialize. You further need to provide two methods:
/// [`Cacheable::commit`] and [`Cacheable::merge_into`].
///
/// Double-buffered state works by having a "pending" instance of your type,
/// into which new values provided by the client are inserted. When the client
/// sends `wl_surface.commit`, the [`Cacheable::commit`] method will be
/// invoked on your value. This method is expected to produce a new instance of
/// your type, that will be stored in the cache, and eventually merged into the
/// current state.
///
/// In most cases, this method will simply produce a copy of the pending state,
/// but you might need additional logic in some cases, such as for handling
/// non-cloneable resources (which thus need to be moved into the produce value).
///
/// Then at some point the [`Cacheable::merge_into`] method of your type will be
/// invoked. In this method, `self` acts as the update that should be merged into
/// the current state provided as argument. In simple cases, the action would just
/// be to copy `self` into the current state, but more complex cases require
/// additional logic.
pub trait Cacheable: Default {
    /// Produce a new state to be cached from the pending state
    fn commit(&mut self, dh: &DisplayHandle) -> Self;
    /// Merge a state update into the current state
    fn merge_into(self, into: &mut Self, dh: &DisplayHandle);
}

/// The commits of one surface that wait at once, each with its own state: those
/// of a synchronized subsurface until its parent commits, and those behind a
/// blocker. A commit past this many joins the newest waiting one and takes
/// effect with it, so the states kept do not grow with the commits sent.
///
/// A commit that is not yet part of a queued transaction (a synchronized
/// subsurface's) does not join a state that a queued transaction names: that
/// transaction would find nothing under the id it holds. It is pushed instead,
/// once for each queued transaction, and later commits join it.
const CACHE_MAX: usize = 8;

/// The same for commits queued with their own transaction, which wait behind
/// blockers only. The compositor bounds those itself: ade lets a fifo barrier
/// go at 32 held commits a surface, each shown for a frame, and a queue of
/// fewer than 33 would show some of them as one.
const CACHE_QUEUED_MAX: usize = 33;

/// The room a queue of waiting states keeps once it is empty.
const CACHE_KEEP: usize = 4;

/// What a commit may do with the newest waiting state when the queue is full.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Join {
    /// The commit's own transaction is queued with it
    pub(crate) queued: bool,
    /// The newest id of this surface that a queued transaction names
    pub(crate) named: Option<Serial>,
}

/// Double buffered cached state of type `T`
#[derive(Debug)]
pub struct CachedState<T> {
    pending: T,
    cache: VecDeque<(Serial, T)>,
    current: T,
}

impl<T: Default> Default for CachedState<T> {
    fn default() -> Self {
        CachedState {
            pending: T::default(),
            cache: VecDeque::new(),
            current: T::default(),
        }
    }
}

impl<T> CachedState<T> {
    /// Access the current state for `T`
    pub fn current(&mut self) -> &mut T {
        &mut self.current
    }

    /// Access the pending state for `T`
    pub fn pending(&mut self) -> &mut T {
        &mut self.pending
    }
}

trait Cache: Downcast {
    fn commit(&self, commit_id: Option<Serial>, join: Join, dh: &DisplayHandle);
    fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle);
}

impl_downcast!(Cache);

impl<T: Cacheable + 'static> Cache for Mutex<CachedState<T>> {
    fn commit(&self, commit_id: Option<Serial>, join: Join, dh: &DisplayHandle) {
        let mut guard = self.lock().unwrap();
        let me = &mut *guard;
        let new_state = me.pending.commit(dh);
        if let Some(id) = commit_id {
            let most = match join.queued {
                true => CACHE_QUEUED_MAX,
                false => CACHE_MAX,
            };
            let full = me.cache.len() >= most;
            let free = |cid: Serial| join.queued || join.named.is_none_or(|named| cid > named);
            match me.cache.back_mut() {
                Some(&mut (cid, ref mut state)) if cid == id => new_state.merge_into(state, dh),
                Some(&mut (ref mut cid, ref mut state)) if full && free(*cid) => {
                    *cid = id;
                    new_state.merge_into(state, dh);
                }
                _ => me.cache.push_back((id, new_state)),
            }
        } else {
            for (_, state) in me.cache.drain(..) {
                state.merge_into(&mut me.current, dh);
            }
            new_state.merge_into(&mut me.current, dh);
            me.cache.shrink_to(CACHE_KEEP);
        }
    }

    fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle) {
        let mut me = self.lock().unwrap();
        loop {
            if me.cache.front().map(|&(s, _)| s > commit_id).unwrap_or(true) {
                // if the cache is empty or the next state has a commit_id greater than the requested one
                break;
            }
            me.cache.pop_front().unwrap().1.merge_into(&mut me.current, dh);
        }
        if me.cache.is_empty() && me.cache.capacity() > CACHE_KEEP {
            me.cache.shrink_to(CACHE_KEEP);
        }
    }
}

/// A typemap-like container for double-buffered values
///
/// All values inserted into this container must implement the [`Cacheable`] trait,
/// which defines their buffering semantics. They furthermore must be [`Send`] as the surface state
/// can be accessed from multiple threads (but [`Sync`] is not required, the surface internally synchronizes
/// access to its state).
///
/// [`MultiCache::get`] provides access to the [`CachedState`] associated with a particular type.
///
/// Consumers of surface state (like compositor applications using Smithay) will mostly be concerned
/// with the [`CachedState::current`] method, which gives access to the current state of the surface for
/// a particular type.
///
/// Writers of protocol extensions logic will mostly be concerned with the [`CachedState::pending`] method,
/// which provides access to the pending state of the surface, in which new state from clients will be
/// stored.
///
/// This contained has [`Mutex`]-like semantics: values of multiple stored types can be accessed at the
/// same time, but accessing the same value multiple times will cause a deadlock.
/// The stored values are initialized lazily the first time [`get`][Self::get] is invoked with this type as argument.
pub struct MultiCache {
    caches: appendlist::AppendList<Box<dyn Cache + Send>>,
    named: Option<Serial>,
}

impl std::fmt::Debug for MultiCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiCache").finish_non_exhaustive()
    }
}

impl MultiCache {
    pub(crate) fn new() -> Self {
        Self {
            caches: appendlist::AppendList::new(),
            named: None,
        }
    }

    fn find_or_insert<T: Cacheable + Send + 'static>(&self) -> &Mutex<CachedState<T>> {
        for cache in &self.caches {
            if let Some(v) = (**cache).as_any().downcast_ref() {
                return v;
            }
        }
        // if we reach here, then the value is not yet in the list, insert it
        self.caches
            .push(Box::new(Mutex::new(CachedState::<T>::default())) as Box<_>);
        (*self.caches[self.caches.len() - 1])
            .as_any()
            .downcast_ref()
            .unwrap()
    }

    /// Access the [`CachedState`] associated with type `T`
    pub fn get<T: Cacheable + Send + 'static>(&self) -> MutexGuard<'_, CachedState<T>> {
        self.find_or_insert::<T>().lock().unwrap()
    }

    /// Check if the container currently contains values for type `T`
    pub fn has<T: Cacheable + Send + 'static>(&self) -> bool {
        self.caches
            .iter()
            .any(|c| (**c).as_any().is::<Mutex<CachedState<T>>>())
    }

    /// Commits the pending state, invoking Cacheable::commit()
    ///
    /// If commit_id is None, then the pending state is directly merged
    /// into the current state. Otherwise, this id is used to store the
    /// cached state. An id ca no longer be re-used as soon as not new id
    /// has been used in between. Provided IDs are expected to be provided
    /// in increasing order according to `Serial` semantics.
    ///
    /// If a None commit is given but there are some cached states, they'll
    /// all be merged into the current state before merging the pending one.
    ///
    /// `queued` says that the transaction of this commit is queued with it; a
    /// commit whose transaction still waits for a parent's is not.
    pub(crate) fn commit(&mut self, commit_id: Option<Serial>, queued: bool, dh: &DisplayHandle) {
        let join = Join {
            queued,
            named: self.named,
        };
        // none of the underlying borrow_mut() can panic, as we hold
        // a &mut reference to the container, non are borrowed.
        for cache in &self.caches {
            cache.commit(commit_id, join, dh);
        }
    }

    /// A queued transaction names the state of this id
    pub(crate) fn name(&mut self, id: Serial) {
        if self.named.is_none_or(|named| id > named) {
            self.named = Some(id);
        }
    }

    /// Apply given identified cached state to the current one
    ///
    /// All other preceding states are applied as well, to preserve commit ordering
    pub(crate) fn apply_state(&self, commit_id: Serial, dh: &DisplayHandle) {
        // none of the underlying borrow_mut() can panic, as we hold
        // a &mut reference to the container, non are borrowed.
        for cache in &self.caches {
            cache.apply_state(commit_id, dh);
        }
    }
}
