#include "joystick.h"

Joystick::Joystick(int _id){
    id = getId(_id);

    update();
}

void Joystick::update(){
    present = glfwJoystickPresent(id);
    if(present){
        name = glfwGetJoystickName(id);
        axes = glfwGetJoystickAxes(id, &axesCount);
        buttons = glfwGetJoystickButtons(id, &buttonCount);
    }
}

float Joystick::axesState(int axis){
    if(present){
        return axes[axis];
    }
    return 0.0f;
}

unsigned char Joystick::buttonState(int button){
    return present ? buttons[button] : GLFW_RELEASE;
}

int Joystick::getAxesCount(){
    return axesCount;
}

int Joystick::getButtonCount(){
    return buttonCount;
}

bool Joystick::isPresent(){
    return present;
}

const char* Joystick::getName(){
    return name;
}

int Joystick::getId(int _id){
    return GLFW_JOYSTICK_1 + _id;
}
