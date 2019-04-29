
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return 'E:\\VOCtrainval_11-May-2012\\VOCdevkit'  # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return '/path/to/SBD/'  # folder with img/, inst/, cls/, etc.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'
