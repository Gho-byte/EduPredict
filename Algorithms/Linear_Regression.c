#include <stdlib.h>
#include <string.h>
#include <stdio.h>


int main()
{
    FILE* file = fopen("/home/mohamed/Documents/Github/Gho-byte/EduPredict/Dataset/Transformed/transformed_dataset.bin", "r");
    if (file == NULL){
        perror("[-] Error While Reading The File");
        return 1;
    }

    char line[40];
    while(fgets(line, sizeof(line), file)){
        line[strcspn(line, "\n")] = 0;
        char* token = strtok(line, ",");
        int loops = 0;
        while (token != NULL){
            loops ++;
            printf("[+] The Token Value is %d: %s\n", loops, token);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
    return 0;
}