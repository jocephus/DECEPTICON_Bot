# DECEPTICON_Bot

If running this on a Jetson Nano, you will need to get their version of Tensorflow 2.20: https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770

Everything else is installed with `python3 -m pip install -r requirements.txt`

## Defining Source Accounts:
You will need to enter accounts enclosed in `'` on line 29. An example is:
`USERS = ['Dc865_owL']`

## About DECEPTICON Bot
When we see the terms Natural Language Processing (NLP) or Machine Learning (ML), often, our guts are correct, and it is vendor marketing material, frequently containing FUD. After tinkering with various libraries in Python and R with the use of some OSINT and SOCMINT techniques, I have found a use for NLP and ML that is 100% FUD free in the form of a brand new, Python-based tool.

In this presentation, which goes further than the previous DECEPTICON presentation, we address topics that I have frequently spoken about in past years is disinformation, deception, OSINT, and OPSEC. When working through learning NLP and ML in Python, it dawned on me: marry these technologies with DECEPTICON for good. Enter the DECEPTICON bot. The DECEPTICON bot is a python* based tool that connects to social media via APIs to read posts/tweets to determine patterns of posting intervals and content then takes over to autonomously post for the user. What is the application you ask: people who are trying to enhance their OPSEC and abandon social media accounts that have been targeted without setting off alarms to their adversaries. Use case scenarios include public figures, executives, and, most importantly – domestic violence and trafficking victims.
