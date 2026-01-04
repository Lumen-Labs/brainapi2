"""
File: /misc.py
Created Date: Saturday January 3rd 2026
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday January 3rd 2026 7:43:36 pm
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

NODE_DESCRIPTION_PROMPT = """
You are an experienced reader and analyst.
You are given a {element_type} to describe, 
your task is to carefully read and understand the related information and provide a concise description of {element_name} (the {element_type}).

The thing to describe:
{element}

Your output must only be the description of the thing above without any other text or comments.

Begin!
"""
