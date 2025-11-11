#ifndef IA_CLIENT_H
#define IA_CLIENT_H

typedef struct {
    char base_url[256];
    int timeout;
} IA_Config;

void ia_init(IA_Config *cfg, const char *url, int timeout);
int ia_search(IA_Config *cfg, const char *query, int limit, char *out, size_t out_size);

#endif