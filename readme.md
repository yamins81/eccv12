
Steps to run random search experiment
=====================================

1. Set up a mongod server
 
    * pick a directory PATH

    * pick a port number PORT

    * run mongod --dbpath=PATH --port=PORT

2. Start a hyperopt random search using mongo
    
    * hyperopt-mongo-search --workdir=~/exp/eccv12/workdir --mongo=HOST:PORT/db eccv12.plugins.Bandit hyperopt.Random

3. Start worker processes to dequeue from that mongo

    * hyperopt-mongo-worker  --mongo=HOST:PORT/db

    -or-

    * THEANO_FLAGS=device=gpu0,floatX=float32 hyperopt-mongo-worker  --mongo=HOST:PORT/db


