import litserve as ls
from aspen_backend.middleware import AspenStreamingAPI

app = AspenStreamingAPI()

if __name__ == "__main__":
    api = app
    server = ls.LitServer(api, stream=True)
    server.run(port=8000)