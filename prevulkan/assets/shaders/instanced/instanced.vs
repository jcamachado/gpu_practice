//These are from buffer
//layout(qualifier1​, qualifier2​ = value, ...) variable definition
//the layout is the location of the vertex attribute in the VBO
layout (location = 0) in vec3 aPos;         // vertex position
layout (location = 1) in vec3 aNormal;      // vertex normal
layout (location = 2) in vec2 aTexCoord;    // vertex texture coordinates
layout (location = 3) in vec3 aTangent;     // vertex tangent
layout (location = 4) in mat4 model;       // model matrix, replaces offSet and size (4x4 occupies 4 memory slots)
layout (location = 8) in mat3 normalModel;  // values in tangent space


// We will keep some data in world space but we will also pass in tangent data
out VS_OUT {     
    vec3 FragPos;   // Available in world space coordinates and in tangent space coordinates
    vec2 TexCoord;

    TangentLights tanLights;
} vs_out;

uniform mat4 view;
uniform mat4 projection;

uniform vec3 viewPos;
uniform bool noNormalMap;

void main() {
    /*
        Get position in world space

        We need to transform the vertex pos to vec4 (4x1) to be able to multiply by the model matrix(4x4)
        Apply model transformation
    */
    vs_out.FragPos = vec3(model * vec4(aPos, 1.0));      
                    

    // Set texture coordinates
    vs_out.TexCoord = aTexCoord;

    // Determine normal vector in tangent space
    if (noNormalMap) {
		// determine normal vector in tangent space
		vs_out.tanLights.Normal = normalize(aNormal);
        vs_out.tanLights.FragPos = vs_out.FragPos;
		vs_out.tanLights.ViewPos = viewPos;
        vs_out.tanLights.dirLightDirection = dirLight.direction;
        		for (int i = 0; i < nPointLights; i++) {
			vs_out.tanLights.pointLightPositions[i] = pointLights[i].position;
		}
        for (int i = 0; i < nSpotLights; i++) {
			vs_out.tanLights.spotLightPositions[i] = spotLights[i].position;
			vs_out.tanLights.spotLightDirections[i] = spotLights[i].direction;
		}
    }
	else
	{
        // determine normal vector in tangent space
		vs_out.tanLights.Normal = normalize(normalModel * aNormal);
        // calculate tangent space matrix
		vec3 T = normalize(normalModel * aTangent);
		vec3 N = vs_out.tanLights.Normal;
		T = normalize(T - dot(T, N) * N); // re-orthogonalize T with respect to N
		vec3 B = cross(N, T); // get B, perpendicular to N and T
		mat3 TBNinv = transpose(mat3(T, B, N)); // orthogonal matrix => transpose = inverse

		// transform positions to the tangent space
		vs_out.tanLights.FragPos = TBNinv * vs_out.FragPos;
		vs_out.tanLights.ViewPos = TBNinv * viewPos;

		// directional light
		vs_out.tanLights.dirLightDirection = TBNinv * dirLight.direction;

		// point lights
		for (int i = 0; i < nPointLights; i++) {
			vs_out.tanLights.pointLightPositions[i] = TBNinv * pointLights[i].position;
		}

		// spot lights
		for (int i = 0; i < nSpotLights; i++) {
			vs_out.tanLights.spotLightPositions[i] = TBNinv * spotLights[i].position;
			vs_out.tanLights.spotLightDirections[i] = TBNinv * spotLights[i].direction;
		}
    }
    // // Calculate tangent space matrix
    // vec3 T = normalize(normalModel * aTangent);
    // vec3 N = vs_out.tanLights.Normal;
    // /*
    //     -Make sure T is perpendicular to N
    //     T is a combination of some T components.
    //         -T is tangent vector, Tt is component in T direction, T^ is T unit vector, Tt*T^ (seems to be the component perpendicular to N)
    //          Tn is the component in N direction(dot(T, N^)), N^ is N unit vector // Question: is is Tn = dot(T, N^) or TnN^ = dot(T, N)
    //     T = Tt * T^ + Tn * N^ 
    //     If we want only the component perpendicular to N (maybe Tt*T^)
    //     T - dot(T, N^) = Tt*T^              
    //     T = T - dot(T, N) * N
    //     dot(T, N) is the component of T that is parallel to N (the magnitude)
    //     and multiply by N to get the component of T that is parallel to N

    //     - To transforming world space coordinates into another space coordinates,
    //     we need to multiply it by the inverse of the space matrix (transformation matrix)
    // */
    // T = normalize(T - dot(T, N) * N); // Re-orthogonalize T with respect to N
    // vec3 B = cross(T, N);           // To make sure B is perpendicular to T and N
    // // TBN is the space of Tangent, Bitangent, Normal (tangent space)
    // mat3 TBNinv = transpose(mat3(T, B, N)); // Orthogonal matrix => transpose = inverse

    // // Transform (world?) positions to the tangent space
    // vs_out.tanLights.FragPos = TBNinv * vs_out.FragPos;
    // vs_out.tanLights.ViewPos = TBNinv * viewPos;

    // // Directional light
    // vs_out.tanLights.dirLightDirection = TBNinv * dirLight.direction;

    // // Point lights
    // // Takes pointLights from the UBO and transform them into tangent space
    // for(int i = 0; i < nPointLights; i++) {
    //     vs_out.tanLights.pointLightPositions[i] = TBNinv * pointLights[i].position;
    // }

    // // Spot lights
    // for(int i = 0; i < nSpotLights; i++) {
    //     vs_out.tanLights.spotLightPositions[i] = TBNinv * spotLights[i].position;
    //     vs_out.tanLights.spotLightDirections[i] = TBNinv * spotLights[i].direction;
    // }

    //fix

    // Set output for fragment shader
    gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}