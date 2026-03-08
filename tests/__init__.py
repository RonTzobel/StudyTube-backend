# tests/
# All automated tests live here.
# Uses pytest as the test runner.
#
# Recommended structure as the project grows:
#   tests/
#     conftest.py          → shared fixtures (test DB session, test client, etc.)
#     test_health.py       → tests for the health check endpoint
#     test_auth.py         → tests for register/login
#     test_videos.py       → tests for video CRUD
