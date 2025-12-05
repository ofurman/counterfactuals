from __future__ import annotations

import tensorflow as tf


def ensure_tf_session() -> tf.compat.v1.Session:
    """Return a TF v1 session and expose backend.get_session for Keras 3 runtimes."""
    backend = tf.compat.v1.keras.backend
    get_session = getattr(backend, "get_session", None)
    if callable(get_session):
        session = get_session()
        if session is not None:
            return session

    session: tf.compat.v1.Session | None = getattr(ensure_tf_session, "_session", None)
    if session is None:
        session = tf.compat.v1.Session()
        setattr(ensure_tf_session, "_session", session)

    setattr(backend, "get_session", lambda: session)
    return session
