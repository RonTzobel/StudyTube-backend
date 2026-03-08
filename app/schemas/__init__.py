# schemas/
# Pydantic schemas used for request validation and response serialization.
# These are NOT database models — they define what the API accepts and returns.
#
# Naming convention:
#   - <Entity>Create  → what the client sends to create a resource
#   - <Entity>Read    → what the API returns to the client
#   - <Entity>Update  → what the client sends to update a resource (all fields optional)
