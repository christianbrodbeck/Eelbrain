'''History for wx GUIs'''
# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>

from .._wxutils import logger


class Action(object):

    def do(self, doc):
        raise NotImplementedError

    def undo(self, doc):
        raise NotImplementedError


class History():
    """The history as a list of action objects

    Public interface
    ----------------
    can_redo() : bool
        Whether the history can redo an action.
    can_undo() : bool
        Whether the history can redo an action.
    do(action)
        perform a action
    is_saved() : bool
        Whether the current state is saved
    redo()
        Redo the latest undone action.
    ...
    """

    def __init__(self, doc):
        self.doc = doc
        self._history = []
        self._saved_change_subscriptions = []
        # point to last executed action (always < 0)
        self._last_action_idx = -1
        # point to action after which we saved ( > 0 if ever saved)
        self._saved_idx = -1

    def can_redo(self):
        return self._last_action_idx < -1

    def can_undo(self):
        return len(self._history) + self._last_action_idx >= 0

    def do(self, action):
        logger.debug("Do action: %s", action.desc)
        was_saved = self.is_saved()
        action.do(self.doc)
        if self._last_action_idx < -1:
            # discard alternate future
            self._history = self._history[:self._last_action_idx + 1]
            self._last_action_idx = -1
            if self._saved_idx >= len(self._history):
                self._saved_idx = -1
        self._history.append(action)
        self._process_saved_change(was_saved)

    def _process_saved_change(self, was_saved):
        """Process a state change in whether all changes are saved

        Parameters
        ----------
        was_saved : bool
            Whether all changes were saved before the current change happened.
        """
        is_saved = self.is_saved()
        if is_saved != was_saved:
            self.doc.saved = is_saved
            for func in self._saved_change_subscriptions:
                func()

    def is_saved(self):
        """Determine whether the document is saved

        Returns
        -------
        is_saved : bool
            Whether the document is saved (i.e., contains no unsaved changes).
        """
        current_index = len(self._history) + self._last_action_idx
        if current_index == -1 and self._saved_idx < 0:
            return True  # no actions and never saved
        return self._saved_idx == current_index

    def redo(self):
        was_saved = self.is_saved()
        if self._last_action_idx == -1:
            raise RuntimeError("We are at the tip of the history")
        action = self._history[self._last_action_idx + 1]
        logger.debug("Redo action: %s", action.desc)
        action.do(self.doc)
        self._last_action_idx += 1
        self._process_saved_change(was_saved)

    def register_save(self):
        "Notify the history that the document is saved at the current state"
        was_saved = self.is_saved()
        self._saved_idx = len(self._history) + self._last_action_idx
        self._process_saved_change(was_saved)

    def subscribe_to_saved_change(self, callback):
        "callback(saved)"
        self._saved_change_subscriptions.append(callback)

    def undo(self):
        was_saved = self.is_saved()
        if -self._last_action_idx > len(self._history):
            raise RuntimeError("We are at the beginning of the history")
        action = self._history[self._last_action_idx]
        logger.debug("Undo action: %s", action.desc)
        action.undo(self.doc)
        self._last_action_idx -= 1
        self._process_saved_change(was_saved)
