import pandas as pd


df = pd.read_csv('./runs/resnet50_skip_after_nonlin_imagenet/metrics.csv')
# ax = df_clssic.plot(x='epoch', y=['train_loss', 'test_loss'], grid=True)
# df_gen = pd.read_csv('generalised_res_net2.csv')
# df = pd.merge(df_clssic, df_gen, on='epoch', suffixes=['_cls', '_gen'])
ax = df.plot(x='epoch', y=['train_loss', 'test_loss'], grid=True)
ax.get_figure().savefig('losses.png')

ax = df.plot(x='epoch', y=['train_acc', 'test_acc'], grid=True)
ax.get_figure().savefig('accuracy.png')
