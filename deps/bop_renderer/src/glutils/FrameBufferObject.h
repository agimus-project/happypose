#pragma once

#include "gl_core_3_3.h"

#include <iostream>
#include <vector>

class FrameBufferObject
{
	public:
		FrameBufferObject();
		~FrameBufferObject();

		bool attachColorBuffer(int width, int height, int format);
		bool detachColorBuffer(int num);
		bool enableColorBuffer(int num = 0);
		bool disableColorBuffer();

		void setDepthBufferEnabled(bool state);

		bool createFBO();
		bool enableFBO();
		bool disableFBO();
		bool destroyFBO();
		bool isCreated();

		int getColorBufferCount();
		int getWidth();
		int getHeight();
		GLuint getColorBufferHandle(int num=0);
        GLuint getDepthBufferHandle();
        
        const std::string getStatusString();

	private:
		bool fboCreated;
		bool depthEnabled;
		int width, height;
		std::vector<GLuint> colorBuffers;
		GLuint depthBuffer;
		GLuint fbo;

		GLsizei oldViewport[4];
		GLint oldFBO;
};
