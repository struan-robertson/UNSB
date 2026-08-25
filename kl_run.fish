#!/bin/fish
cd ~/Development/Doctorate/siamese
for r in gan_santa_kl_1 gan_santa_kl_2 gan_santa_kl_3
    for f in checkpoints/$r/best/siamese_*.tar
        ln -sf ../(basename $f) $f
    end
    for p in 5 10
        uv run python src/evaluation.py checkpoints/$r/best config.toml --dataset wvu -p $p --log-name eval_wvu_p$p.log
    end
end
