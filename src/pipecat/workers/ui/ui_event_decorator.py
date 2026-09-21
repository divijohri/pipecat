#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The ``@ui_event`` decorator, which lives in :mod:`pipecat.workers.ui_event_decorator`."""

from pipecat.workers.ui_event_decorator import _collect_ui_event_handlers, ui_event

__all__ = ["_collect_ui_event_handlers", "ui_event"]
