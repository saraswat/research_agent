# How was this repo created.

First, I specify the problem to Gemini 2.5. My prompt:


> Design for me an agent infrastructure, using OpenAI Agents library, that accepts as input any user research problem definition and returns results of the same quality as Google Gemini Deep Research, using a web-search agent as tool, and other tools as appropriate. It needs to be able to generate citations for all claims in the final report.

The result is in claude_design.md.

I then copied this over into Claude Desktop, using Sonnet 3.7. And told it:

> Write the code for this system.

The code in the first commit is the code produced by Claude. 
