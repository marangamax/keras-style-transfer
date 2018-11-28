Neural Style Transfer
---------------
"In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image."

This repository contains a command-line tool for performing neural style transfer between any content and style image of choice, letting a machine do the composing instead of a human.

Neural style transfer is quite a well-known concept amongst machine learning researchers by now and has also been revealed to the public by various companies that attempt to monetize on this new creative dimension.
Unfortunately, I observed there isn't much in between research and the existing applications that is useable by the wider public. There are some really cool possibilities using this technology, so I thought that 
this should be changed. It follows that I created this so the artists, creative visionaries, and those simply 
looking to pimp their selfies can also try it out.. and feel like hackers at the same. 


How to use
---------------
1. Choose a content image and a style image and put them somewhere on your computer and remember the path to each file.
1. Open up a Terminal window and go to the repository (will be different if you installed it somewhere other than your Desktop):

    `cd Desktop/keras-style-transfer`
    
3. Now we're ready to run the program:

    `python3 src/run.py --style='path/to/style_image' --content='path/to/content_image' --save_path='path/to/saved_image'`
    
    An example helps, let's say I have a content image called content.jpg and a style image called style.jpg, both located
    on my Desktop and I want to save the output to my Desktop:
    
    `python3 src/run.py --style='/Users/Max/Desktop/style.jpg' --content='/Users/Max/Desktop/content.jpg' --save_path='/Users/Max/Desktop/output.jpg'`
    
It is normal that this takes some time on a MacBook, usually ~15 minutes depending on the size of your CPU.

4. At this point,



Non-Coder Installation Procedure
---------------
I'm an OSX user so unfortunately this will not work for Windows (Linux should be fine). Don't let this scare you away, it won't take more than 10 minutes.
1. Open up Terminal
2. Check that you have python3 installed:

    `python3 --version`
    
If the output is something like '3.x.x', continue to the next step. If 'command not found', follow the instructions at this page https://wsvincent.com/install-python3-mac/.
3.  Next write the following command to install this repository to your Desktop (or anywhere else if you know what your doing):

    `cd Desktop`
    
    then
    
    `git clone https://github.com/marangamax/keras-style-transfer`
    
4. Almost there, now we just have to install the dependencies and your ready to go:

    `cd keras-style-transfer`
    
    then 
    
    `pip3 install -r requirements`