#include "FrameBufferObject.h"

FrameBufferObject::FrameBufferObject()
{
	fboCreated = false;
	depthEnabled = true;
	width = height = -1;
}

FrameBufferObject::~FrameBufferObject()
{
	if (fboCreated)
		destroyFBO();
}

bool FrameBufferObject::attachColorBuffer(int width, int height, int format)
{
//	unsigned char pixels[width*height*3];
//	for(unsigned int y = 0; y < height; y++) {
//		for(unsigned int x = 0; x < width; x++) {
//            unsigned int i = 3 * (y * width + x);
//            pixels[i] = x % 255;
//            pixels[i + 1] = 0;
//            pixels[i + 2] = 0;
//		}
//	}

    GLuint fboColor;
	glGenTextures(1, &fboColor);
	glBindTexture(GL_TEXTURE_2D, fboColor);
//	glTexImage2D(GL_TEXTURE_2D, 0, format,  width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	glTexImage2D(GL_TEXTURE_2D, 0, format,  width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
//	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

//	glBindTexture(GL_TEXTURE_2D, 0);

	colorBuffers.push_back(fboColor);

	this->width = width;
	this->height = height;

	return (glGetError() == GL_NO_ERROR);
}

bool FrameBufferObject::detachColorBuffer(int num)
{
	if (num < 0 || num >= static_cast<int>(colorBuffers.size()))
		return false;

	glDeleteTextures(1, &(colorBuffers[num]));

	return true;
}

bool FrameBufferObject::enableColorBuffer(int num )
{
	if (num < 0 || num>=static_cast<int>(colorBuffers.size()))
		return false;

	glBindTexture(GL_TEXTURE_2D, colorBuffers[num]);

	return true;
}

bool FrameBufferObject::disableColorBuffer()
{
	glBindTexture(GL_TEXTURE_2D, 0);
	return true;
}

void FrameBufferObject::setDepthBufferEnabled(bool state)
{
	depthEnabled = state;
}

bool FrameBufferObject::createFBO()
{
	if (colorBuffers.empty())
        return true;

	if (fboCreated)
		destroyFBO();


    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &oldFBO);

	// create the frame buffer
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	// add a depth buffer if needed
	if (depthEnabled) {
		glGenRenderbuffers(1, &depthBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
	}

	// attach all color buffers
	for (unsigned int i=0; i<colorBuffers.size(); ++i) {
//		glBindTexture(GL_TEXTURE_2D, handle);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, colorBuffers[i], 0);
	}

	// all done, check status
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);

	fboCreated = (status == GL_FRAMEBUFFER_COMPLETE);

	if (!fboCreated)
		std::cout<<"Framebuffer completion failed!"<<std::endl;

    glBindFramebuffer(GL_FRAMEBUFFER, oldFBO);

	return true;

}

const std::string FrameBufferObject::getStatusString() 
{
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &oldFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, oldFBO);
    
    switch(status) {
        case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return "Not all attachments were properly created";
/*        case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return "Dimensions of attachments differ in size"; */
        case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return "Framebuffer does not contain attachments";
        case GL_FRAMEBUFFER_UNSUPPORTED:
            return "Combination of internal formats not supported";
    }
    
    return "No error";
}

bool FrameBufferObject::enableFBO()
{
	if (!fboCreated && !createFBO())
		return false;

	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &oldFBO);
	glGetIntegerv(GL_VIEWPORT, oldViewport);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glViewport(0, 0, width, height);

	return true;
}

bool FrameBufferObject::disableFBO()
{
	glViewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
	glBindFramebuffer(GL_FRAMEBUFFER, oldFBO);

	return true;
}

bool FrameBufferObject::destroyFBO()
{
	if (!fboCreated)
		return true;

	glDeleteFramebuffers(1, &fbo);
	glDeleteRenderbuffers(1, &depthBuffer);

	glDeleteTextures(colorBuffers.size(), colorBuffers.data());

	colorBuffers.clear();

	fboCreated = false;

	return true;
}

bool FrameBufferObject::isCreated()
{
	return fboCreated;
}


int FrameBufferObject::getColorBufferCount()
{
	return colorBuffers.size();
}

int FrameBufferObject::getWidth()
{
	return width;
}

int FrameBufferObject::getHeight()
{
	return height;
}

GLuint FrameBufferObject::getColorBufferHandle(int num)
{
	if (num < 0 || num >= static_cast<int>(colorBuffers.size()))
		return 0;

	return colorBuffers[num];
}

GLuint FrameBufferObject::getDepthBufferHandle()
{
    return depthBuffer;
}
