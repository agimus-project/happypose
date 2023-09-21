 # Download example data for minimal testing

```
cd $MEGAPOSE_DATA_DIR
wget https://memmo-data.laas.fr/static/examples.tar.xz
tar xf examples.tar.xz
```
 
 # Download pre-trained pose estimation models

Download pose estimation models to $MEGAPOSE_DATA_DIR/megapose-models:

```
python -m happypose.toolbox.utils.download --megapose_models
```