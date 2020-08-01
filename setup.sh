apt-get update -y; apt-get upgrade -y; apt-get dist-upgrade -y; apt autoremove -y;
apt-get install python3 python3-pip git python3-venv;
python3 -m pip install --no-use-wheel --no-cache-dir Cython;
python3 -m pip install -U -r requirements.txt;
