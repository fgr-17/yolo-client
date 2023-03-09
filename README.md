# Python Template

Yolo client prepared to receive frames and process the image, plot boxes, and generate a new frame.

## Author: fede, peter

## Installation

Must have docker and docker compose installed.

Build and get inside the container executing:

```bash
docker-compose up -d
docker exec -it yolo bash
```

## Usage

Go to the python app folder and start the main.py directly. The app listens to the port specified in the code (4008 now)

```bash
cd /workspace/src
./main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)