import torch
import torch.nn as nn
import torch.optim as optim


def make_G_mlp(input_dim, output_dim, hidden_dim=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, output_dim)
    )


def make_D_mlp(input_dim, hidden_dim=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim, 1)
    )


def block():
    pass


class Reshape(nn.Module):

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(self.target_shape)


def make_G_cnn(channels_noise, channels_img, features_g=16):
    return nn.Sequential(

        # Input: N x channels_noise

        Reshape(target_shape=(-1, channels_noise, 1, 1)),

        # img: N x channel_noise x 1 x 1

        nn.ConvTranspose2d(
            channels_noise, features_g * 16, 4, 1, 0, bias=False,  # stride doens't matter here
        ),
        nn.BatchNorm2d(features_g * 16),
        nn.ReLU(),  # 4x4x256

        nn.ConvTranspose2d(
            features_g * 16, features_g * 8, 4, 2, 1, bias=False,
        ),
        nn.BatchNorm2d(features_g * 8),
        nn.ReLU(),  # 8x8x128

        nn.ConvTranspose2d(
            features_g * 8, features_g * 4, 4, 2, 1, bias=False,
        ),
        nn.BatchNorm2d(features_g * 4),
        nn.ReLU(),  # 16x16x64

        nn.ConvTranspose2d(
            features_g * 4, features_g * 2, 4, 2, 1, bias=False,
        ),
        nn.BatchNorm2d(features_g * 2),
        nn.ReLU(),  # 32x32x32

        nn.ConvTranspose2d(
            features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
        ),   # 64x64x3

        nn.Tanh(),
    )


def make_D_cnn(channels_img, features_d=16):
    return nn.Sequential(

        # input: 64x64x3

        nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),  # 32x32x16

        # block 1
        nn.Conv2d(
            features_d, features_d * 2, 4, 2, 1, bias=False,
        ),
        nn.InstanceNorm2d(features_d * 2, affine=True),
        nn.LeakyReLU(0.2),  # 16x16x32

        # block 2
        nn.Conv2d(
            features_d * 2, features_d * 4, 4, 2, 1, bias=False,
        ),
        nn.InstanceNorm2d(features_d * 4, affine=True),
        nn.LeakyReLU(0.2),  # 8x8x64

        # block 3
        nn.Conv2d(
            features_d * 4, features_d * 8, 4, 2, 1, bias=False,
        ),
        nn.InstanceNorm2d(features_d * 8, affine=True),
        nn.LeakyReLU(0.2),  # 4x4x128

        # After all _block img output is 4x4 (Conv2d below makes into 1x1)

        nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # stride doesn't matter here
        Reshape(target_shape=(-1, 1))

    )


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def gradient_penalty(critic, mixed, use_custom):
    # Calculate critic scores
    mixed_scores = critic(mixed)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    if use_custom:
        gp = torch.mean((nn.functional.relu(gradient_norm - 1)) ** 2)
    else:
        gp = torch.mean((gradient_norm - 1) ** 2)

    return gp


def wasserstein_distance(x_from_p_data, x_from_p_G, f):
    # sample-based approximation to Wasserstein distance based on Kantorovich-Rubinstein theorem
    return torch.mean(f(x_from_p_data)) - torch.mean(f(x_from_p_G))


class WGAN_GP:

    """
    Implements Wasserstein GAN with gradient penalty (WGAN-GP).

    Methods are designed for neural networks (generator and critic) to interact with outside world
    either by (1) generating new data from generator or (2) updating generator / critic.

    This class does NOT contain the training loop.
    """

    def __init__(
            self,
            G_input_dim,
            G_output_shape,
            gp_lamb=10,
            use_custom_gp=True,
            lr=1e-4,
            adam_betas=(0, 0.9),
    ):

        assert len(G_output_shape) == 1 or len(G_output_shape) == 3  # vector or image

        self.G_input_dim = G_input_dim
        self.G_output_shape = G_output_shape
        self.gp_lamb = gp_lamb
        self.use_custom_gp = use_custom_gp
        self.lr = lr
        self.adam_betas = adam_betas

        if len(G_output_shape) == 3:  # instantiate W-DCGAN architecture

            self.G = None
            self.D = make_G_cnn(channels_img=G_output_shape)

            initialize_weights(self.G)
            initialize_weights(self.D)

        else:  # instantiate standard WGAN architecture

            self.G = make_G_mlp(input_dim=G_input_dim, output_dim=G_output_shape[0])
            self.D = make_D_mlp(input_dim=G_output_shape[0])

        self.G_optimizer = optim.Adam(self.G.parameters(), betas=self.adam_betas, lr=self.lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), betas=self.adam_betas, lr=self.lr)

    def sample_G(self, batch_size):
        noise_vecs = torch.randn(batch_size, self.G_input_dim)  # standard Gaussian
        gens = self.G(noise_vecs)
        return gens.view(batch_size, *self.G_output_shape)

    def update_D_one_iter(self, real_data_batch):

        bs = real_data_batch.shape[0]

        fake_data_batch = self.sample_G(batch_size=bs)

        assert real_data_batch.shape == (bs, *self.G_output_shape)
        assert fake_data_batch.shape == (bs, *self.G_output_shape)

        random_nums_batch = torch.rand(bs, 1)  # will be broaccasted to (bs, self.G_output_dim)

        mixed_batch = real_data_batch * random_nums_batch + fake_data_batch * (1 - random_nums_batch)

        assert mixed_batch.shape == (bs, *self.G_output_shape)

        w = wasserstein_distance(x_from_p_data=real_data_batch, x_from_p_G=fake_data_batch, f=self.D)
        gp = gradient_penalty(self.D, mixed_batch, use_custom=self.use_custom_gp)

        assert w.shape == ()
        assert gp.shape == ()

        objective_to_maximize = w - self.gp_lamb * gp  # maximize distance while staying close to <=1-Lipschitz
        loss = - objective_to_maximize

        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()

        return float(w)

    def update_G_one_iter(self, batch_size):

        fake_data_batch = self.sample_G(batch_size=batch_size)

        objective_to_maximize = torch.mean(self.D(fake_data_batch))
        loss = - objective_to_maximize

        self.G_optimizer.zero_grad()
        loss.backward()
        self.G_optimizer.step()
