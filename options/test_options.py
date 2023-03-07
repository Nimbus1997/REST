from email.policy import default
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        #ellen_100 sample 뽑아서 보기 - 안좋은 이미지 중에서 각각의 이미지 종류 비율 학인하기 위해서 
        parser.add_argument('--result_sample', type = int, default=0, help='set it to 1 to sample 100 images to see the ratio of each images')
        parser.add_argument('--ellen_test', action='store_true', help='only save fakeB')
        
        self.isTrain = False
        return parser
