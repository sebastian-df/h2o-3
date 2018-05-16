from __future__ import print_function
from builtins import range
import sys, os
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def prostate_gbm():
  # Connect to a pre-existing cluster
  # connect to localhost:54321

  df = h2o.import_file(path=pyunit_utils.locate("smalldata/logreg/prostate.csv"))
  df.describe()

  # Remove ID from training frame
  train = df.drop("ID")

  # For VOL & GLEASON, a zero really means "missing"
  vol = train['VOL']
  vol[vol == 0] = None
  gle = train['GLEASON']
  gle[gle == 0] = None

  # Convert CAPSULE to a logical factor
  train['CAPSULE'] = train['CAPSULE'].asfactor()

  # See that the data is ready
  train.describe()
  x= list(range(1, train.ncol))
  params = {'ntrees':50, 'learn_rate':0.1, 'distribution':"bernoulli"}
  my_gbm = pyunit_utils.build_save_model_GBM(params, x, train, "CAPSULE")
  MOJONAME = pyunit_utils.getMojoName(my_gbm._id)
  TMPDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath('__file__')), "..", "results", MOJONAME))

  h2o.download_csv(train[x], os.path.join(TMPDIR, 'in.csv'))  # save test file, h2o predict/mojo use same file
  pred_h2o, pred_mojo = pyunit_utils.mojo_predict(my_gbm, TMPDIR, MOJONAME)  # load model and perform predict

  my_gbm.show()





if __name__ == "__main__":
  pyunit_utils.standalone_test(prostate_gbm)
else:
  prostate_gbm()
