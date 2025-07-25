
PATH=/workspace-SR004.nfs2/d.tarasov/envs/dtarasov-speech2latex/bin:$PATH

WORKDIR=/workspace-SR004.nfs2/d.tarasov/rsi-speech2latex/MathSpeech/ASR

cd $WORKDIR

python ASR_s2l_equations.py $@


