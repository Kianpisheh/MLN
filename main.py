from pracmln import MLN
from pracmln import Database
from pracmln import MLNQuery, MLNLearn
from pracmln.utils.project import PRACMLNConfig


mln = MLN(mlnfile="./activity.mln")
db = Database.load(mln, "./activity.db")
config1 = PRACMLNConfig("./learn.conf")

# inference
learner = MLNLearn(mln=mln, db=db, config=config1, verbose=True)
learner.run()
x = 1
