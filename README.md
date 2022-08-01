# arxiv-reader - Converting arXiv papers to audio.
## Dr. Tristan Behrens, [LinkedIn](https://www.linkedin.com/in/dr-tristan-behrens-734967a2/).

Reads arXiv papers using Text-to-Speech. Uses the Facebook model `facebook/fastspeech2-en-ljspeech`.

## Installation.

Install these requirements:

```
pip install transformers
pip install g2p_en
pip install beautifulsoup4
pip install arxiv-downloader
```

The script also requires pandoc. If you are running on a Mac with Homebrew installed, you can install pandoc with:

```
brew install pandoc
```

If you do not run a Mac with Homebrew - which is a reason for regret - follow the installation instructions for your OS.

## How to run.

If you want to convert "Attention is all you need" to speech, you can run the following command:

```
python convert.py "1706.03762"
```

Better use the quotes around the arXiv ID.

## Issues and features.

Feel free to add more features and improve things. Pull requests are more than welcome. And do not hesitate GitHub issues.

Have fun!