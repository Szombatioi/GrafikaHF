vec3 hyNormalize(vec3 vector){

}

// 1. Egy irányra merőleges irány állítása.

// 2. Adott pontból és sebesség vektorral induló pont helyének és sebesség vektorának számítása t idővel később.
vec3 hyMoveP(vec3 point, vec3 vector, float t){
    return point * coshf(t) + hyNormalize(vector) * sinhf(t);
}
vec3 hyMoveV(vec3 point, vec3 vector, float t){
    return hyNormalize(point * sinhf(t) + hyNormalize(vector) * sinhf(t));
}

// 3. Egy ponthoz képest egy másik pont irányának és távolságának meghatározása.
// 4. Egy ponthoz képest adott irányban és távolságra lévő pont előállítása.
// 5. Egy pontban egy vektor elforgatása adott szöggel.
// 6. Egy közelítő pont és sebességvektorhoz a geometria szabályait teljesítő, közeli pont és sebesség választása.