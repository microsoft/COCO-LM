pip install --user --editable .
pip install --user sentencepiece
if [  -d  fused_ops ]
then
    pip install --user --editable fused_ops
fi
