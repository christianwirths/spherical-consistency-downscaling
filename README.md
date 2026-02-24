# Generative climate model downscaling for the Bern3D EMIC

## References
The implementation is mostly based on the repositories:

- https://github.com/p-hss/consistency-climate-downscaling/
- https://github.com/yang-song/score_sde_pytorch 
- https://github.com/openai/consistency_models.

as well as the papers:

- P. Hess et al., 2025: https://www.nature.com/articles/s42256-025-00980-5
- Y. Song et al., 2021: https://arxiv.org/abs/2011.13456
- Y. Song et al., 2023: https://arxiv.org/abs/2303.01469
- Bischoff and Deck, 2023: https://arxiv.org/abs/2305.01822
- T. Karras et al., 2022: https://arxiv.org/abs/2206.00364


python main.py --name "era5_ens" -dm "consistency" --use_ema --n_worker 8 --n_epochs 50 --batch_size 1