# core/
# Cross-cutting concerns that don't belong to a single feature.
# Good candidates for this folder:
#   - security.py     → password hashing, JWT creation/verification
#   - dependencies.py → shared FastAPI dependencies (e.g. get_current_user)
#   - exceptions.py   → custom exception classes and HTTP error handlers
#   - logging.py      → structured logging setup
#
# Currently empty — will grow as auth and other shared utilities are added.
