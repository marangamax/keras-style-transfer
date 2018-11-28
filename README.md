Neural Style Transfer
---------------
"In fine art, especially painting, humans have mastered the skill to create unique visual experiences through composing a complex interplay between the content and style of an image."

This repository contains a command-line tool for performing neural style transfer between any content and style image of choice, letting a machine do the composing instead of a human.

Neural style transfer is quite a well-known concept amongst machine learning researchers by now and has also been revealed to the public by various companies that attempt to monetize on this new creative dimension.
Unfortunately, I observed there isn't much in between research and the existing applications that is useable by the wider public. There are some really cool possibilities using this technology, so I thought that 
this should be changed. It follows that I created this so the artists, creative visionaries, and those simply 
looking to pimp their selfies can also try it out.. and feel like hackers at the same. 


Installation Procedure
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

4. So now you've run the machine using your images, but maybe they're not exactly what you expected and you want to adjust them. Maybe you want to see more content or style, or you just want to 
see how abstract you can get. Fortunately, there are several parameters you can pass to the command:

    `--size is the image size, default is 256. At the moment the maximum I can get to is 1024 with a 16 GB CPU`
    
    `--iter is the number of iterations, default is 10. Increasing this will basically let your machine go further down into the rabbit hole of abstraction. After 20 there's not much noticeable difference`
    
    `--alpha is the ratio of the content to the style image, default is 0.01. Increase this for more content, decrease this for more style`
    
    `--afactor is the abstraction factor, default is 3. This values ranges from 1 to 5, where 5 is the most abstract`
    
    So using this, let's tune my example command a bit more:
    
    `python3 src/run.py --style='/Users/Max/Desktop/style.jpg' --content='/Users/Max/Desktop/content.jpg' --save_path='/Users/Max/Desktop/output.jpg' --size=512 --alpha=0.001 --afactor=5`
    
All this can lead to some wonderfully interesting results, hope you enjoy it as much as I do and please share any issues but also interesting results!