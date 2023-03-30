﻿//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Szombati Olivér
// Neptun : P37PLU
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

//TODO: kommentek törlése, nyilatkozat elolvasása, források megjelölése
//TODO: vetített koordináták fv., azokat kell a vectorba tenni
//TODO: vetítés z koordinátája?
//TODO: a körök megjelenítése legyen simán euklideszi? Elvégre nem mosódhatnak el a körök



/*
	1. Egy irányra merőleges irány állítása.
	2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
	3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
	4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
	5. Egy pontban egy vektor elforgatása adott szöggel.
	6. Egy közelítő pont és sebességvektorhoz a Geometria szabályait teljesítő, közeli pont és sebesség választása.
*/

//1.
vec3 perpendicular(vec3 vector) {
	vec3 temp(0, 0, -1);
	vector.z *= -1;
	return cross(vector, temp);
}

//2.
vec3 movePoint(vec3 point, vec3 vector, float delta_t) {
	vec3 temp = vec3(point * cosh(delta_t) + normalize(vector) * sinh(delta_t));
//	printf("%.2f %.2f %.2f\n", temp.x, temp.y, temp.z);
	return temp;
}

vec3 moveVec(vec3 point, vec3 vector, float delta_t){
	return vec3(point * sinh(delta_t) + normalize(vector) * cosh(delta_t));
}

//3. ?


//4. TODO: szemek, szájak rajzolásához 
vec3 pointByDirAndDist(vec3 point, vec3 dir, float dist) {
	return  point + normalize(dir)*dist;
}

//5. 
vec3 rotateVector(vec3 vector, float angle) {
	return vec3(vector * cos(angle) + perpendicular(vector) * sin(angle));
}

//6. ?
//-------------------------

class Renderer : public GPUProgram {
	
	const char* const vertexSource = R"(
		#version 330
		precision highp float;
		layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

		void main() { gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); }	
	)";

	const char* const fragmentSource = R"(
		#version 330
		precision highp float;
		uniform vec3 color;
		out vec4 fragmentColor;	

		void main() { fragmentColor = vec4(color, 1); }
	)";

	unsigned int vao, vbo;
public:
	Renderer() {
		glViewport(0, 0, windowWidth, windowHeight);

		create(vertexSource, fragmentSource, "outColor");
		glGenVertexArrays(1, &vao); 
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 		
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	std::vector<vec2> project(const std::vector<vec3> points) { //TODO
		std::vector<vec2> projected;
		int i = 0;
		for (auto& point : points) {
			projected.push_back(vec2(point.x / (point.z + 1.0), point.y / (point.z + 1.0)));
		}
		return projected;
	}

	vec2 project(const vec3 point) {
		vec2 res = vec2(point.x / (point.z + 1.0), point.y / (point.z + 1.0));
		return res;
	}

	void DrawGPU(int type, std::vector<vec2> vertices, vec3 color) {
		setUniform(color, "color");
		glBindVertexArray(vao);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), &vertices[0], GL_STATIC_DRAW);
		glDrawArrays(type, 0, vertices.size());
	}

	void DrawGPU(int type, std::vector<vec3> vertices, vec3 color) {
		DrawGPU(type, project(vertices), color);
	}

	~Renderer() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

Renderer* renderer;
const int nOfCirclePoints = 30;
long t;

//for tests
//bool validPoint(vec3 point) { return point.x* point.x + point.y* point.y - point.z* point.z == -1; }
//bool validVector(vec3 vector, vec3 point) { return point.x*vector.x + point.y * vector.y + point.z * vector.z == 0; }

class Circle {
	vec3 centerPoint;
	//rip circlePoints
	float radius;
public:
	Circle(vec3 c = vec3(0,0,0), float r = 1) : centerPoint(c), radius(r) {
		/*for (int i = 0; i < nOfCirclePoints; i++) {
			float phi = i * 2.0f * M_PI / nOfCirclePoints;
			float x = centerPoint.x + cosf(phi) * radius;
			float y = centerPoint.y + sinf(phi) * radius;
			circlePoints.push_back(vec3(x, y, centerPoint.z)); //TODO
		}*/
	}
	void draw(vec3 colors) {
		vec3 angle;
		if() angle = vec3(1,0,0); //ha a centertől 0.001-re van
		else angle = vec3(0,0,1); //ha nem centerbe van, akkor vector: pos->0,0,1
		std::vector<vec3> points;
		//vec3 start = centerPoint;
		//perpendic normalise(cross())
		
		for(float i = 0; i < nOfCirclePoints; i++){
//			float angle = képlet
			//new point, dir
			//push back
			start = movePoint(centerPoint, angle, i);
			points.push_back(start);
			moveVec(centerPoint, angle, i);
		}
		
		renderer->DrawGPU(GL_TRIANGLE_FAN, points, colors);
	}
	float getR() {
		return radius;
	}
};

class Hami {
protected:
	std::vector<vec2> slimePoints;
	vec3 position, direction;
	Circle centerCircle, leftEye, rightEye, leftPupil, rightPupil, mouth;
public:
	Hami() {}
	Hami(vec3 start, vec3 dir) : centerCircle(start, 0.2f), direction(dir) {
		//leftEye = Circle(pointByDirAndDist(position, rotateVector(direction, 0.01), centerCircle.getR()), 0.1);
	}

	virtual void move() { 
		//slimePoints.push_back(renderer->project(position)); //TODO: check if contains
	}
	virtual void draw(vec3 color){
		animateMouth();
		centerCircle.draw(color);
		//renderer->DrawGPU(GL_POINTS, slimePoints, vec3(1, 1, 1));
		/*leftEye.draw(vec3(1,1,1));
		rightEye.draw(vec3(1, 1, 1));
		rightPupil.draw(vec3(1, 1, 1));
		leftPupil.draw(vec3(1, 1, 1));
		mouth.draw(vec3(0,0,0));*/
	}

	void rotate(bool right, float phi = 0.1f) {
		direction = rotateVector(direction, right ? phi : -phi);
	}

	void animateMouth() {

	}
};

class PirosHami : public Hami {
public:
	PirosHami() {}
	PirosHami(vec3 start, vec3 dir) : Hami(start, dir) {}
	void move() { 
		position = movePoint(position, direction, 0.01);
		direction = moveVec(position, direction, 0.01);
		//printf("%.2f %.2f %.2f\n", position.x, position.y, position.z);
	}
};

class ZoldHami : public Hami {
public:
	ZoldHami() {}
	ZoldHami(vec3 start, vec3 dir) : Hami(start, dir) {}
	void move() {
		//Hami::move();
	}
};

Circle center;
PirosHami pirosHami;
ZoldHami zoldHami;

//testing the projection with lines
//std::vector<vec3> line;

void onInitialization() {
	
	//printf("(2,2,3) %s\t(2,-2,0) %s\n", validPoint(vec3(0,0,1)) ? "valid" : "invalid", validVector(vec3(1,1,0), vec3(0, 0, 1)) ? "valid" : "invalid");

	renderer = new Renderer();

/*	glPointSize(10.0f);  //////////test: a vetítésem jó
	line.push_back(vec3(1, 1, sqrt(3)));
	line.push_back(vec3(0,0,1));
	line.push_back(vec3(2,2,3));
	line.push_back(vec3(3,3,sqrt(19)));*/
	//center = Circle();
	pirosHami = PirosHami(vec3(0, 0, 1), vec3(1,1,0));
	zoldHami = ZoldHami(vec3(2, 2, 3), vec3(2, -2, 0));

	
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	//center.draw(vec3(0.5f, 0.5f, 0.5f));
	pirosHami.draw(vec3(1,0,0));
	zoldHami.draw(vec3(0,1,0));

//	renderer->DrawGPU(GL_POINTS, line, vec3(1,1,1)); //test

	//zoldHami.move(); //TODO: lehet nem ide kéne tenni
	glutSwapBuffers();
}

bool keys[256];
void onKeyboard(unsigned char key, int pX, int pY) {
	keys[key] = true;
	/*switch (key) {
	case 'e':
		printf("Pressed key: e, going towards.\n");
		pirosHami.move();
		break;
	case 's':
		printf("Pressed key: s, rotating to left.\n");
		pirosHami.rotate(true);
		break;
	case 'f':
		printf("Pressed key: f, rotating to right.\n");
		pirosHami.rotate(false);
		break;
	}*/
}

void onIdle() {
	t = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	if(keys['e']) pirosHami.move();
/*	if(keys['s']) printf("Rotate left\n");
	if(keys['f']) printf("Rotate right\n");*/
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
	keys[key] = false;
}
void onMouseMotion(int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
