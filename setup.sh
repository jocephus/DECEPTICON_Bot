apt-get update -y; apt-get upgrade -y; apt-get dist-upgrae -y; apt automremove -y;
apt-get install python3 python3-pip git;
pip3 install --no-use-wheel --no-cache-dir Cython;
pip3 install -U -r requirements.txt
