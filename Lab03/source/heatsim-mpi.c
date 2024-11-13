#include <assert.h>
#include <stddef.h>

#include "heatsim.h"
#include "log.h"

typedef struct grid_parameters {
    unsigned int width;
    unsigned int height;
    unsigned int padding;
} grid_param_t;


int heatsim_init(heatsim_t* heatsim, unsigned int dim_x, unsigned int dim_y) {
    /*
     * TODO: Initialiser tous les membres de la structure `heatsim`.
     *       Le communicateur doit être périodique. Le communicateur
     *       cartésien est périodique en X et Y.
     */
    int dims[2] = {dim_x, dim_y};
    int periods[2] = {1, 1};  // Periodic in both dimensions

    // Create cartesian communicator
    if (MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &heatsim->communicator) != MPI_SUCCESS) {
        goto fail_exit;
    }

    // Get rank and size
    if (MPI_Comm_rank(heatsim->communicator, &heatsim->rank) != MPI_SUCCESS) {
        goto fail_exit;
    }
    if (MPI_Comm_size(heatsim->communicator, &heatsim->rank_count) != MPI_SUCCESS) {
        goto fail_exit;
    }

    // Get coordinates
    if (MPI_Cart_coords(heatsim->communicator, heatsim->rank, 2, heatsim->coordinates) != MPI_SUCCESS) {
        goto fail_exit;
    }

    // Get neighbor ranks
    if (MPI_Cart_shift(heatsim->communicator, 0, 1, &heatsim->rank_west_peer, &heatsim->rank_east_peer) != MPI_SUCCESS) {
        goto fail_exit;
    }
    if (MPI_Cart_shift(heatsim->communicator, 1, 1, &heatsim->rank_north_peer, &heatsim->rank_south_peer) != MPI_SUCCESS) {
        goto fail_exit;
    }

    return 0;
    fail_exit:
    return -1;
}

int heatsim_send_grids(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Envoyer toutes les `grid` aux autres rangs. Cette fonction
     *       est appelé pour le rang 0. Par exemple, si le rang 3 est à la
     *       coordonnée cartésienne (0, 2), alors on y envoit le `grid`
     *       à la position (0, 2) dans `cart`.
     *
     *       Il est recommandé d'envoyer les paramètres `width`, `height`
     *       et `padding` avant les données. De cette manière, le receveur
     *       peut allouer la structure avec `grid_create` directement.
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée.
     */

    MPI_Datatype param_type;
    {
        int blocklengths[3] = {1, 1, 1};
        MPI_Aint displacements[3];
        MPI_Datatype types[3] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};

        displacements[0] = offsetof(grid_param_t, width);
        displacements[1] = offsetof(grid_param_t, height);
        displacements[2] = offsetof(grid_param_t, padding);

        MPI_Type_create_struct(3, blocklengths, displacements, types, &param_type);
        MPI_Type_commit(&param_type);
    }

    for (int rank = 1; rank < heatsim->rank_count; rank++) {
        int coords[2];
        MPI_Cart_coords(heatsim->communicator, rank, 2, coords);
        grid_t* grid = cart2d_get_grid(cart, coords[0], coords[1]);

        // Send grid parameters
        grid_param_t params = {grid->width, grid->height, grid->padding};
        MPI_Request req;
        MPI_Isend(&params, 1, param_type, rank, 0, heatsim->communicator, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        // Send grid data
        MPI_Isend(grid->data, grid->width * grid->height, MPI_DOUBLE, rank, 1, heatsim->communicator, &req);
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }

    MPI_Type_free(&param_type);
    return 1;
}

grid_t* heatsim_receive_grid(heatsim_t* heatsim) {
    /*
     * TODO: Recevoir un `grid ` du rang 0. Il est important de noté que
     *       toutes les `grid` ne sont pas nécessairement de la même
     *       dimension (habituellement ±1 en largeur et hauteur). Utilisez
     *       la fonction `grid_create` pour allouer un `grid`.
     *
     *       Utilisez `grid_create` pour allouer le `grid` à retourner.
     */

    MPI_Datatype gridParamType;
    MPI_Datatype paramFieldTypes[3] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_UNSIGNED};
    int paramFieldLength[3] = {1, 1, 1};
    MPI_Aint paramFieldPosition[3];

    // Get offsets for the grid parameters
    grid_param_t temp_param;
    paramFieldPosition[0] = offsetof(grid_param_t, width);
    paramFieldPosition[1] = offsetof(grid_param_t, height);
    paramFieldPosition[2] = offsetof(grid_param_t, padding);

    MPI_Type_create_struct(3, paramFieldLength, paramFieldPosition,
                           paramFieldTypes, &gridParamType);
    MPI_Type_commit(&gridParamType);

    // Receive grid parameters
    grid_param_t params;
    MPI_Status status;
    MPI_Recv(&params, 1, gridParamType, 0, 0, heatsim->communicator, &status);

    // Create grid with received parameters
    grid_t* grid = grid_create(params.width, params.height, params.padding);
    if (!grid) {
        goto fail_exit;
    }

    // Receive grid data
    MPI_Recv(grid->data, grid->width * grid->height, MPI_DOUBLE, 0, 1, heatsim->communicator, &status);

    MPI_Type_free(&gridParamType);
    return grid;

    fail_exit:
        MPI_Type_free(&gridParamType);
        return NULL;
}

int heatsim_exchange_borders(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */

    // Special case for single process
    if (heatsim->rank_count == 1) {
        for (int i = 0; i < grid->width; i++) {
            *grid_get_cell(grid, i, -1) = *grid_get_cell(grid, i, grid->height-1);
            *grid_get_cell(grid, i, grid->height) = *grid_get_cell(grid, i, 0);
        }
        for (int j = 0; j < grid->height; j++) {
            *grid_get_cell(grid, -1, j) = *grid_get_cell(grid, grid->width-1, j);
            *grid_get_cell(grid, grid->width, j) = *grid_get_cell(grid, 0, j);
        }
        return 1;
    }

    // Create MPI type for north/south borders (contiguous)
    MPI_Datatype border_ns;
    MPI_Type_contiguous(grid->width, MPI_DOUBLE, &border_ns);
    MPI_Type_commit(&border_ns);

    // Create MPI type for east/west borders (vector)
    MPI_Datatype border_ew;
    MPI_Type_vector(grid->height, 1, grid->width + 2, MPI_DOUBLE, &border_ew);
    MPI_Type_commit(&border_ew);

    // Send to north, receive from south
    MPI_Send(grid_get_cell(grid, 0, 0), 1, border_ns,heatsim->rank_north_peer, 0, heatsim->communicator);
    MPI_Recv(grid_get_cell(grid, 0, -1), 1, border_ns,heatsim->rank_south_peer, 0, heatsim->communicator, MPI_STATUS_IGNORE);

    // Send to south, receive from north
    MPI_Send(grid_get_cell(grid, 0, grid->height-1), 1, border_ns,heatsim->rank_south_peer, 0, heatsim->communicator);
    MPI_Recv(grid_get_cell(grid, 0, grid->height), 1, border_ns,heatsim->rank_north_peer, 0, heatsim->communicator, MPI_STATUS_IGNORE);

    // Send to west, receive from east
    MPI_Send(grid_get_cell(grid, 0, 0), 1, border_ew, heatsim->rank_west_peer, 0, heatsim->communicator);
    MPI_Recv(grid_get_cell(grid, -1, 0), 1, border_ew, heatsim->rank_east_peer, 0, heatsim->communicator, MPI_STATUS_IGNORE);

    // Send to east, receive from west
    MPI_Send(grid_get_cell(grid, grid->width-1, 0), 1, border_ew, heatsim->rank_east_peer, 0, heatsim->communicator);
    MPI_Recv(grid_get_cell(grid, grid->width, 0), 1, border_ew,
             heatsim->rank_west_peer, 0, heatsim->communicator, MPI_STATUS_IGNORE);

    // Clean up MPI types
    MPI_Type_free(&border_ns);
    MPI_Type_free(&border_ew);

    return 1;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */

    assert(grid->padding == 0);

    // Create MPI type for grid data
    MPI_Datatype gridDataType;
    int dataLength = grid->width * grid->height;
    MPI_Type_contiguous(dataLength, MPI_DOUBLE, &gridDataType);
    MPI_Type_commit(&gridDataType);

    // Send grid data to rank 0
    MPI_Send(grid->data, 1, gridDataType, 0, 10, heatsim->communicator);

    MPI_Type_free(&gridDataType);
    return 1;
}

int heatsim_receive_results(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Recevoir toutes les `grid` des autres rangs. Aucune `grid`
     *       n'a de rembourage (padding = 0).
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée
     *       qui va recevoir le contenue (`data`) d'un autre noeud.
     */
    MPI_Status status;
    int coords[2];
    grid_t* grid;

    // Create MPI type for grid data
    MPI_Datatype gridDataType;

    // Receive from all ranks except 0
    for (int rank = 1; rank < heatsim->rank_count; rank++) {
        // Get coordinates for this rank
        MPI_Cart_coords(heatsim->communicator, rank, 2, coords);

        // Get the grid at these coordinates
        grid = cart2d_get_grid(cart, coords[0], coords[1]);
        if (!grid) {
            goto fail_exit;
        }

        // Create type for this grid's data
        int dataLength = grid->width * grid->height;
        MPI_Type_contiguous(dataLength, MPI_DOUBLE, &gridDataType);
        MPI_Type_commit(&gridDataType);

        // Receive grid data
        MPI_Recv(grid->data, 1, gridDataType, rank, 10, heatsim->communicator, &status);

        MPI_Type_free(&gridDataType);
    }

    return 1;

    fail_exit:
        return -1;
}
