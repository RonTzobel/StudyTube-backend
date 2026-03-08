# services/
# Business logic lives here — between the router (HTTP layer) and the database.
# Routers call services. Services talk to the database via the session.
# This keeps route handlers clean and makes logic easy to test independently.
