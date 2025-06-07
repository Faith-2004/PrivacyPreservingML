import flwr as fl
from flwr.server import ServerConfig

fl.server.start_server(
     config=fl.server.ServerConfig(num_rounds=5)
)
