#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

class OmpRealize {
public:
    int siz;
    unsigned short int Count_threads;
    unsigned char* Img;
    double epsilon;
    OmpRealize(unsigned char* img, int n,  unsigned short int count_threads, double coef) {

        Img = img;
        siz = n;
        Count_threads = count_threads;
        epsilon = coef;
    }
    void single()
    {
        unsigned int* Gist = new(nothrow) unsigned int[256];
        memset(Gist, 0, 1024);
        unsigned char min, max;
        int count = ceil(siz * epsilon);

        for (int i = 0; i < siz; i++)
        {
            try
            {
                Gist[Img[i]] += 1;
            }
            catch (const std::exception&)
            {
                cerr << "error: Couldn't read the pixel in gistogram" << endl;
                delete[] Gist;
                delete[] Img;
                exit(1);
            }
        }

        for (int i = 0; i < 256; i++)
        {
            if (Gist[i] < count)
            {
                count -= Gist[i];
                
            }
            else if (Gist[i] == count)
            {
                min = i+1;
                break;
            }
            else 
            {
                min = i;
                break;
            }
        }
        count = ceil(siz * epsilon);
        
        for (int i = 255; i > -1; i--)
        {
            if (Gist[i] < count)
            {
                count -= Gist[i];

            }
            else if (Gist[i] == count)
            {
                max = i + 1;
                break;
            }
            else
            {
                max = i;
                break;
            }
        }

        for (int i = 0; i < siz; i++)
        {
            
            if ((Img[i] - min) *255 / (max - min) < 0) {
                Img[i] = 0;
            }
            else if ((Img[i] - min) * 255 / (max - min) > 255) {
                Img[i] = 255;
            }
            else {
                Img[i] = (Img[i] - min) * 255 / (max - min);
            }
        }
    }

    void paral(string type)
    {
        unsigned char minimum, maximum;
        unsigned int* Gist = new(nothrow) unsigned int[256];
        memset(Gist, 0, 1024);
        int count = ceil(siz * epsilon);
        if (type == "P5")
        {
#pragma omp parallel num_threads(Count_threads)
            {
                unsigned int* Gist_t = new(nothrow) unsigned int[256];
                memset(Gist_t, 0, 1024);
#pragma omp for schedule(guided,1000)
                for (int i = 0; i < siz; i++)
                {
                    try
                    {
                        Gist_t[Img[i]] += 1;
                    }
                    catch (const std::exception&)
                    {
                        cerr << "error: Couldn't read the pixel in gistogram" << endl;
                        delete[] Gist_t;
                        delete[] Gist;
                        delete[] Img;
                        exit(1);
                    }
                }
                for (int i = 0; i < 256; i++)
                {
#pragma omp atomic
                    Gist[i] += Gist_t[i];
                }
            }
            for (int i = 0; i < 256; i++)
            {
                if (Gist[i] < count)
                {
                    count -= Gist[i];

                }
                else if (Gist[i] == count)
                {
                    minimum = i + 1;
                    break;
                }
                else
                {
                    minimum = i;
                    break;
                }
            }
            count = ceil(siz * epsilon);

            for (int i = 255; i > -1; i--)
            {
                if (Gist[i] < count)
                {
                    count -= Gist[i];

                }
                else if (Gist[i] == count)
                {
                    maximum = i + 1;
                    break;
                }
                else
                {
                    maximum = i;
                    break;
                }
            }
        }
        else
        {
#pragma omp parallel num_threads(Count_threads)
            {
                unsigned int* Gist_t_1 = new(nothrow) unsigned int[256];
                unsigned int* Gist_t_2 = new(nothrow) unsigned int[256];
                unsigned int* Gist_t_3 = new(nothrow) unsigned int[256];
                memset(Gist_t_1, 0, 1024);
                memset(Gist_t_2, 0, 1024);
                memset(Gist_t_3, 0, 1024);
#pragma omp for schedule(guided,1000)
                for (int i = 0; i < siz; i+=3)
                {
                    try
                    {
                        Gist_t_1[Img[i]] += 1;
                        Gist_t_2[Img[i+1]] += 1;
                        Gist_t_3[Img[i+2]] += 1;
                    }
                    catch (const std::exception&)
                    {
                        cerr << "error: Couldn't read the pixel in gistogram" << endl;
                        delete[] Gist_t_1;
                        delete[] Gist_t_2;
                        delete[] Gist_t_3;
                        delete[] Gist;
                        delete[] Img;
                        exit(1);
                    }
                }
                unsigned int help = 1;
                for (int i = 0; i < 256; i++)
                {
                    help = max({ Gist_t_1[i] ,Gist_t_2[i] , Gist_t_3[i]});
#pragma omp atomic
                    Gist[i] += help;
                }
            }


            for (int i = 0; i < 256; i++)
            {
                if (Gist[i] < count)
                {
                    count -= Gist[i];

                }
                else if (Gist[i] == count)
                {
                    minimum = i + 1;
                    break;
                }
                else
                {
                    minimum = i;
                    break;
                }
            }
            count = ceil(siz * epsilon);

            for (int i = 255; i > -1; i--)
            {
                if (Gist[i] < count)
                {
                    count -= Gist[i];

                }
                else if (Gist[i] == count)
                {
                    maximum = i + 1;
                    break;
                }
                else
                {
                    maximum = i;
                    break;
                }
            }
        }
#pragma omp parallel for schedule(guided,1000) num_threads(Count_threads)
        {
            for (int i = 0; i < siz; i++)
            {

                if ((Img[i] - minimum) * 255 / (maximum - minimum) < 0) {
                    Img[i] = 0;
                }
                else if ((Img[i] - minimum) * 255 / (maximum - minimum) > 255) {
                    Img[i] = 255;
                }
                else {
                    Img[i] = (Img[i] - minimum) * 255 / (maximum - minimum);
                }
            }
        }
    }


    
};


int main(int argc, char* argv[])
{
    unsigned short int  realize;
    unsigned int n, max_lvl;
    int count_treads;
    double coef;

    if (argc != 5) {
        cerr << "error: invalid number of arguments" << endl;
        return 1;
    }
    else {
        try
        {
            count_treads = stoi(argv[3]);
            if (count_treads == 0)
                count_treads = omp_get_max_threads();
            coef = atof(argv[4]);

        }
        catch (const std::exception&)
        {
            cerr << "error: The passed value is not a number" << endl;
            return 1;
        }
    }


    ifstream file(argv[1], ios::binary);
    if (!file.is_open()) {
        cerr << "error: Could not open the file in" << endl;
        return 1;
    
    }
    string type, k;
    int width, height, seez;
    getline(file, type);
    file >> width >> height;
    file >> k;
    file.get();
    
    if (type == "P5")
    {
        seez = width * height;
        
    }
    else
    {
        seez = width * height * 3;
    }
    unsigned char* img = new(nothrow) unsigned char[seez];
    
    for (int i=0;i< seez;i++)
    {
        try
        {
            img[i] = file.get();
        }
        catch (const std::exception&)
        {
            cerr << "error: Couldn't read the pixel" << endl;
            delete[] img;
            return 1;
        }
        
    }
    file.close();
    double start, end;

    OmpRealize omp = OmpRealize(img, seez, count_treads,coef);
    if (count_treads == -1)
    {
        start = omp_get_wtime();
        omp.single();
        end = omp_get_wtime();
        printf("Time (%i thread(s)): %.7f ms\n", 1, (end - start) * 1000);
    }
    else 
    {
        start = omp_get_wtime();
        omp.paral(type);
        end = omp_get_wtime();
        printf("Time (%i thread(s)): %.7f ms\n", count_treads, (end - start) * 1000);
    }
    
    ofstream file_out(argv[2], ios::binary);
    if (!file_out.is_open())
    {
        cerr << "error: Could not open the file out" << endl;
        delete[]img;
        return 1;
    }
    file_out << type << '\n' << width << ' ' << height<<'\n' << k << '\n';
    try
    {
        file_out.write((char*)&img[0], seez);
    }
    catch (const std::exception&)
    {
        cerr << "error: failed to write to file" << endl;
        delete[] img;
        return 1;
    }
    file_out.close();
    delete[]img;
    return 0;
}