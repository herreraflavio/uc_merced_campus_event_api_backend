steps:
1:
create virtual environment
run: python3 -m venv myvenv

2:
make sure to activiate using:
source ./myvenv/bin/activate

3:
download requirements:
pip install -r requirements.txt

4:
make sure to create and load in environment variables into .env
touch .env

5:
add your secretes:
OPENAI_API_KEY=
MONGODB_URI=

6:
right click on your python environment directory and press:
copy path

7:
to select your python interpreter in vscode press the following command:
shift + control + p

enter:
python: select interpreter
press:
enter interpreter path...
and paste in your python environment path

8:
to start script run:
python3 main.py

common issue:
Address already in use

change port to one that is not in use like port 8080
go to main.py, at the bottom, you will see the port number hard coded
rerun script

9:
to test health of server hit the following endpoint after successfully starting it:
http://127.0.0.1:8080/health

should return :
{"message":"Service is up and running","status":"healthy"}
